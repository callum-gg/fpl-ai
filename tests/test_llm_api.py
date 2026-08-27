"""Layers 7 and 8 — the LLM layer and the API surface. docs/12.

Never assert on model prose. Assert on structure, on the guardrails, and on the fact
that the read-only promises are actually enforced by code.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from fplai.db.engine import query_one, writer

# ══ Layer 7 — the LLM layer ══════════════════════════════════════════════════


def test_chat_tools_are_all_read_only():
    """docs/12: a test asserts the tool registry exposes no mutating function."""
    from fplai.llm.chat import assert_read_only

    assert assert_read_only() == []


def test_chat_tool_schemas_match_the_registry():
    from fplai.llm.chat import TOOL_SCHEMAS, TOOLS

    declared = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert declared == set(TOOLS), declared.symmetric_difference(set(TOOLS))
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_every_task_has_a_prompt_and_sane_limits():
    from fplai.llm.tasks import TASKS

    for name, task in TASKS.items():
        assert task.name == name
        assert task.system, f"{name} has no system prompt"
        assert 0.0 <= task.temperature <= 1.0
        assert task.max_tokens > 0


def test_extraction_prompt_states_its_hard_rules():
    """The rules that keep the evidence panel honest must be in the prompt, not folklore."""
    from fplai.llm.tasks import EXTRACT_CLAIMS_SYSTEM

    lowered = EXTRACT_CLAIMS_SYSTEM.lower()
    assert "verbatim" in lowered
    assert "15 words" in lowered
    assert "never infer" in lowered
    assert '"claims": []' in EXTRACT_CLAIMS_SYSTEM   # returning nothing must be allowed
    assert "is_reported" in EXTRACT_CLAIMS_SYSTEM


def test_extraction_drops_claims_whose_span_is_not_in_the_chunk(seeded_season):
    """A hallucinated quote in the evidence panel is worse than no evidence at all."""
    from fplai.llm.extract import _store

    chunk = {"raw_doc_id": _make_raw_doc(), "id": None, "text": "Haaland is fit and will start.",
             "start_s": None}
    written = _store(
        chunk,
        [
            {"player": "Haaland", "claim_type": "injury", "stance": "positive",
             "sentiment": 0.5, "confidence": 0.9, "text_span": "Haaland is fit"},
            {"player": "Haaland", "claim_type": "injury", "stance": "negative",
             "sentiment": -0.9, "confidence": 0.9,
             "text_span": "Haaland tore his hamstring in a fire"},   # never said
        ],
        "test-model", seeded_season, 1,
    )
    assert written == 1


def test_extraction_truncates_overlong_spans(seeded_season):
    from fplai.llm.extract import MAX_SPAN_WORDS, _store

    long_text = " ".join(f"word{i}" for i in range(40))
    doc_id = _make_raw_doc(long_text)
    _store(
        {"raw_doc_id": doc_id, "id": None, "text": long_text, "start_s": None},
        [{"player": None, "claim_type": "form", "stance": "neutral", "sentiment": 0,
          "confidence": 0.5, "text_span": long_text}],
        "test-model", seeded_season, 1,
    )
    row = query_one("SELECT text_span FROM claims WHERE raw_doc_id=?", (doc_id,))
    assert row is not None
    assert len(row["text_span"].split()) <= MAX_SPAN_WORDS


def test_extraction_rejects_unknown_claim_types(seeded_season):
    from fplai.llm.extract import _store

    doc_id = _make_raw_doc("Some text about a player being good.")
    written = _store(
        {"raw_doc_id": doc_id, "id": None, "text": "Some text about a player being good.",
         "start_s": None},
        [{"player": None, "claim_type": "vibes", "text_span": "being good"}],
        "test-model", seeded_season, 1,
    )
    assert written == 0


def _make_raw_doc(text: str = "Haaland is fit and will start.") -> int:
    from fplai.connectors.base import RawDoc, archive

    with writer() as conn:
        doc_id, _ = archive(conn, "rss_news", RawDoc("article", text, url=f"http://x/{hash(text)}"),
                            None)
    return doc_id


def test_adjustment_is_capped_even_when_the_signal_is_absurd(seeded_season, monkeypatch):
    """docs/08: |adjustment| <= min(2.0, 0.25 x base). The leash is code, not a prompt."""
    from fplai.llm import adjust

    fake_claims = [
        {"id": i, "grp": i, "claim_type": "injury", "sentiment": -1.0, "confidence": 1.0,
         "trust_weight": 5.0, "is_reported": 0, "source_id": "rss_news",
         "text_span": "he is finished"}
        for i in range(10)
    ]
    monkeypatch.setattr(adjust, "recent_claims", lambda pid, since_hours=36: fake_claims)

    for base in (1.0, 4.0, 20.0, 100.0):
        value, reason, ids = adjust.compute_adjustment(1, base)
        assert abs(value) <= min(2.0, 0.25 * base) + 1e-9, base
        assert reason and ids


def test_adjustment_needs_two_independent_claims_or_one_tier_one(seeded_season, monkeypatch):
    from fplai.llm import adjust

    one_weak = [{"id": 1, "grp": 1, "claim_type": "injury", "sentiment": -1.0,
                 "confidence": 1.0, "trust_weight": 0.8, "is_reported": 0,
                 "source_id": "reddit", "text_span": "heard he's out"}]
    monkeypatch.setattr(adjust, "recent_claims", lambda pid, since_hours=36: one_weak)
    assert adjust.compute_adjustment(1, 6.0) == (0.0, None, [])

    one_tier1 = [{**one_weak[0], "trust_weight": 2.0, "source_id": "twitter_scrape"}]
    monkeypatch.setattr(adjust, "recent_claims", lambda pid, since_hours=36: one_tier1)
    value, reason, _ = adjust.compute_adjustment(1, 6.0)
    assert value != 0.0 and reason


def test_adjustment_layer_can_be_switched_off(seeded_season, monkeypatch):
    from fplai.db import settings_store
    from fplai.llm import adjust

    monkeypatch.setattr(
        adjust, "global_settings", lambda: {**settings_store.global_settings(),
                                            "adjustment.enabled": False}
    )
    assert adjust.compute_adjustment(1, 6.0) == (0.0, None, [])


def test_llm_json_parsing_survives_fenced_and_chatty_responses():
    from fplai.llm.client import _parse_json

    assert _parse_json('```json\n{"claims": []}\n```') == {"claims": []}
    assert _parse_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}
    assert _parse_json("[1, 2, 3]") == [1, 2, 3]
    assert _parse_json("no json at all") is None
    assert _parse_json("") is None


def test_prompt_hash_is_stable_and_content_sensitive():
    from fplai.llm.client import prompt_hash

    a = [{"role": "user", "content": "hello"}]
    b = [{"role": "user", "content": "hello"}]
    c = [{"role": "user", "content": "goodbye"}]
    assert prompt_hash("chat", "m", a) == prompt_hash("chat", "m", b)
    assert prompt_hash("chat", "m", a) != prompt_hash("chat", "m", c)
    assert prompt_hash("chat", "m", a) != prompt_hash("other", "m", a)


def test_llm_unavailable_is_raised_not_swallowed_silently():
    import asyncio

    from fplai.llm.client import LLMUnavailable, complete
    from fplai.llm.tasks import get

    with pytest.raises(LLMUnavailable):
        asyncio.run(complete(get("chat"), [{"role": "user", "content": "hi"}]))


# ══ Layer 8 — the API surface ════════════════════════════════════════════════


@pytest.fixture(scope="module")
def client():
    from fplai.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_health_reports_the_real_system_state(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["connectors"] >= 20
    assert isinstance(body["llm_configured"], bool)
    assert body["fpl_write_enabled"] is False       # must never default to on


def test_core_read_endpoints_respond(client):
    for path in (
        "/api/squads",
        "/api/sources",
        "/api/settings/global",
        "/api/settings/schema",
        "/api/gameweeks",
        "/api/gameweeks/current",
        "/api/players",
        "/api/fixtures",
        "/api/teams",
        "/api/jobs",
        "/api/models",
        "/api/backtests",
        "/api/pundits",
        "/api/entity-review",
        "/api/llm/usage",
        "/api/feed",
    ):
        assert client.get(path).status_code == 200, path


def test_settings_schema_drives_the_ui(client):
    """The settings UI renders from this, so adding a backend setting needs no frontend
    change. Every entry therefore needs a type and a default."""
    schema = client.get("/api/settings/schema").json()
    assert schema["global"] and schema["squad"]
    for group in ("global", "squad"):
        for field in schema[group]:
            assert field["key"]
            assert field["type"] in ("boolean", "number", "list", "object", "string")
            assert "default" in field
    risk = next(f for f in schema["squad"] if f["key"] == "risk")
    assert risk["widget"] == "slider" and risk["min"] == -1 and risk["max"] == 1


def test_squad_lifecycle(client):
    created = client.post("/api/squads", json={"name": "Test Squad"}).json()
    squad_id = created["id"]
    assert created["name"] == "Test Squad"
    assert created["colour"]

    patched = client.patch(f"/api/squads/{squad_id}", json={"settings": {"risk": -0.8}}).json()
    assert patched["settings"]["risk"] == -0.8

    clone = client.post("/api/squads", json={"name": "Clone", "clone_from": squad_id}).json()
    assert clone["settings"]["risk"] == -0.8      # cloning copies settings

    assert client.delete(f"/api/squads/{squad_id}").status_code == 200
    assert client.get(f"/api/squads/{squad_id}").json()["archived"] == 1


def test_unknown_squad_returns_404(client):
    assert client.get("/api/squads/999999").status_code == 404


def test_settings_patch_round_trips(client):
    client.patch("/api/settings/global", json={"values": {"adjustment.max_points": 1.25}})
    body = client.get("/api/settings/global").json()
    assert body["settings"]["adjustment.max_points"] == 1.25


def test_env_secrets_are_redacted_in_the_settings_view(client):
    """01-architecture: never log or serve secrets."""
    env = client.get("/api/settings/global").json()["env"]
    secrets = [e for e in env if e["secret"]]
    assert secrets
    for entry in secrets:
        assert entry["value"] in (None, "***set***")
        assert entry["read_only"] is True


def test_push_is_refused_when_writing_is_disabled(client):
    squad = client.post("/api/squads", json={"name": "Push Test"}).json()
    res = client.post(
        f"/api/squads/{squad['id']}/push/execute",
        json={"recommendation_id": 1, "confirmation_text": "nope", "dry_run": False},
    )
    assert res.status_code == 403
    assert "refused" in json.dumps(res.json()).lower()


def test_sources_report_why_they_are_unavailable(client):
    sources = client.get("/api/sources").json()
    keyed = [s for s in sources if s["requires_keys"] and not s["available"]]
    for s in keyed:
        assert s["unavailable_reason"]
        assert any(k in s["unavailable_reason"] for k in s["requires_keys"])


def test_model_guard_endpoint_reports_text_dominance(client):
    """docs/05 section F: no text feature may be top-3 by importance."""
    body = client.get("/api/models/guards").json()
    assert body["text_features_not_dominant"] is True
    assert body["offenders"] == []


def test_openapi_schema_generates(client):
    """The frontend types are generated from this, so it must always be valid."""
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "FPL AI"
    paths = schema["paths"]
    for expected in ("/api/squads", "/api/players", "/api/settings/global", "/api/health"):
        assert expected in paths, expected
