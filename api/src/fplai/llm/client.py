"""One OpenAI-compatible client. docs/08.

`base_url` + `api_key` from .env, so OpenRouter, an NVIDIA build endpoint, or a local
Ollama all work identically. Per-task model selection lives in global settings and is
editable in the UI; a model prefixed `alt:` routes to the second endpoint, `ollama:` to
the local one.

Every call is logged to `llm_calls` with a prompt hash. Cacheable tasks return the
cached row on an exact hit, so re-running extraction over unchanged text costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field

import httpx

from ..config import get_settings
from ..db.engine import query_one, writer
from ..db.settings_store import global_settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No endpoint configured. Callers degrade rather than crash."""


@dataclass
class LLMTask:
    name: str
    temperature: float = 0.2
    max_tokens: int = 2000
    response_schema: type | None = None
    cacheable: bool = True
    system: str = ""
    json_only: bool = True

    def model(self) -> str:
        cfg = global_settings().get("llm.tasks", {}).get(self.name, {})
        return cfg.get("model") or get_settings().llm_default_model

    def temp(self) -> float:
        cfg = global_settings().get("llm.tasks", {}).get(self.name, {})
        return float(cfg.get("temperature", self.temperature))


@dataclass
class LLMResponse:
    text: str
    data: dict | list | None = None
    model: str = ""
    cached: bool = False
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)


def _endpoint(model: str) -> tuple[str, str, str]:
    """(base_url, api_key, model_name) after resolving the alt:/ollama: prefixes."""
    s = get_settings()
    if model.startswith("alt:"):
        return s.llm_alt_base_url, s.llm_alt_api_key, model[4:]
    if model.startswith("ollama:"):
        return s.ollama_base_url, s.ollama_api_key, model[7:]
    if s.ollama_enabled and not s.llm_api_key:
        return s.ollama_base_url, s.ollama_api_key, model
    return s.llm_base_url, s.llm_api_key, model


def prompt_hash(task: str, model: str, messages: list[dict]) -> str:
    blob = json.dumps({"task": task, "model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cached(task: str, model: str, phash: str) -> LLMResponse | None:
    if not get_settings().llm_cache_enabled:
        return None
    row = query_one(
        "SELECT response_json FROM llm_calls WHERE task=? AND model=? AND prompt_hash=? AND ok=1",
        (task, model, phash),
    )
    if row is None:
        return None
    payload = json.loads(row["response_json"])
    return LLMResponse(text=payload.get("text", ""), data=payload.get("data"),
                       model=model, cached=True)


def _log_call(task, model, phash, usage, latency_ms, cached, ok, error, response) -> None:
    with writer() as conn:
        conn.execute(
            "INSERT INTO llm_calls(task,model,prompt_hash,prompt_tokens,completion_tokens,"
            "cost_usd,latency_ms,cached,ok,error_text,response_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
            (
                task, model, phash, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                usage.get("cost", 0.0), latency_ms, int(cached), int(ok), error,
                json.dumps(response) if response else None,
            ),
        )


async def complete(
    task: LLMTask, messages: list[dict], tools: list[dict] | None = None, stream: bool = False
) -> LLMResponse:
    model = task.model()
    if not model:
        raise LLMUnavailable(
            f"no model configured for task {task.name!r} — set LLM_DEFAULT_MODEL or "
            f"Settings → Global → LLM"
        )
    base_url, api_key, model_name = _endpoint(model)
    if not base_url:
        raise LLMUnavailable("no LLM endpoint configured")

    if task.system:
        messages = [{"role": "system", "content": task.system}, *messages]

    phash = prompt_hash(task.name, model, messages)
    if task.cacheable and not tools:
        hit = _cached(task.name, model, phash)
        if hit is not None:
            return hit

    body: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": task.temp(),
        "max_tokens": task.max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    elif task.json_only and task.response_schema is not None:
        body["response_format"] = {"type": "json_object"}

    s = get_settings()
    started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(s.llm_max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=s.llm_timeout_s) as client:
                r = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}",
                             "Content-Type": "application/json"},
                    json=body,
                )
                r.raise_for_status()
                data = r.json()
            break
        except Exception as e:
            last_error = e
            if attempt == s.llm_max_retries:
                _log_call(task.name, model, phash, {}, int((time.monotonic() - started) * 1000),
                          False, False, str(e)[:500], None)
                raise
    else:  # pragma: no cover - loop always breaks or raises
        raise last_error or RuntimeError("llm call failed")

    latency = int((time.monotonic() - started) * 1000)
    choice = data["choices"][0]["message"]
    text = choice.get("content") or ""
    usage = data.get("usage", {})

    parsed = None
    if task.response_schema is not None or task.json_only:
        parsed = _parse_json(text)
        if parsed is None and task.response_schema is not None:
            # One repair attempt, then mark it failed and move on. Never loop.
            parsed = await _repair(task, text, base_url, api_key, model_name)

    if choice.get("tool_calls"):
        parsed = {"tool_calls": choice["tool_calls"]}

    response = LLMResponse(text=text, data=parsed, model=model, usage=usage)
    _log_call(task.name, model, phash, usage, latency, False, True, None,
              {"text": text, "data": parsed})
    return response


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_json(text: str):
    if not text:
        return None
    candidates = [text]
    m = _JSON_FENCE.search(text)
    if m:
        candidates.insert(0, m.group(1))
    for c in candidates:
        c = c.strip()
        for start, end in (("{", "}"), ("[", "]")):
            if start in c and end in c:
                snippet = c[c.index(start): c.rindex(end) + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
    return None


async def _repair(task: LLMTask, text: str, base_url: str, api_key: str, model_name: str):
    """One repair attempt. If it still fails, the caller marks the item failed."""
    try:
        async with httpx.AsyncClient(timeout=get_settings().llm_timeout_s) as client:
            r = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON. No prose."},
                        {"role": "user", "content": f"Fix this into valid JSON:\n\n{text[:4000]}"},
                    ],
                    "temperature": 0,
                    "max_tokens": 2000,
                },
            )
            r.raise_for_status()
            return _parse_json(r.json()["choices"][0]["message"].get("content") or "")
    except Exception:  # noqa: BLE001 - repair is best-effort by design
        log.info("json repair failed for task %s", task.name)
        return None


def available() -> bool:
    s = get_settings()
    return bool((s.llm_api_key and s.llm_default_model) or s.ollama_enabled)


def usage_summary(days: int = 30) -> list[dict]:
    """Per-task cost and volume — powers GET /api/llm/usage. If extract_claims is eating
    £4/day you see it here and can switch that task to Ollama in one dropdown."""
    from ..db.engine import query

    return [
        dict(r)
        for r in query(
            "SELECT task, model, COUNT(*) calls, SUM(cached) cached_calls, "
            "SUM(COALESCE(prompt_tokens,0)) prompt_tokens, "
            "SUM(COALESCE(completion_tokens,0)) completion_tokens, "
            "SUM(COALESCE(cost_usd,0)) cost_usd, AVG(latency_ms) avg_latency_ms, "
            "SUM(CASE WHEN ok=0 THEN 1 ELSE 0 END) failures "
            "FROM llm_calls WHERE created_at > datetime('now', ?) "
            "GROUP BY task, model ORDER BY cost_usd DESC",
            (f"-{days} day",),
        )
    ]
