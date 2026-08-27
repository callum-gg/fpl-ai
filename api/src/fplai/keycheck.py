"""Live verification of configured API keys/credentials.

`connectors.registry.status()` and the .env settings view only check that a key is
*present*. This makes one cheap, real call per credentialed service and reports whether
it actually authenticates — a rotated or revoked key still shows "present".

Keys with no consuming integration yet (BETFAIR_*, APIFY_TOKEN, SUPADATA_API_KEY,
REDDIT_CLIENT_*, BLUESKY_*) are skipped: nothing in the app calls them, so there is
nothing to verify.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .config import get_settings
from .connectors.odds_api import BASE as ODDS_API_BASE
from .llm.client import _endpoint

log = logging.getLogger(__name__)


async def _probe(client: httpx.AsyncClient, key: str, service: str, method: str, url: str,
                  **kw) -> dict:
    try:
        r = await client.request(method, url, **kw)
        ok = r.status_code < 400
        detail = f"HTTP {r.status_code}" + (" — key rejected" if r.status_code in (401, 403) else "")
    except httpx.HTTPError as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    return {"key": key, "service": service, "ok": ok, "detail": detail}


async def _check_fpl_login() -> dict:
    from .fplsync.sync import PushRefused, login

    try:
        client = await login()
        await client.aclose()
        return {"key": "fpl_email/fpl_password", "service": "FPL account login",
                 "ok": True, "detail": "login ok"}
    except PushRefused as e:
        return {"key": "fpl_email/fpl_password", "service": "FPL account login",
                 "ok": False, "detail": str(e)}
    except Exception as e:  # noqa: BLE001 - report, don't crash the check
        return {"key": "fpl_email/fpl_password", "service": "FPL account login",
                 "ok": False, "detail": f"{type(e).__name__}: {e}"}


async def verify_all() -> list[dict]:
    """Ping every wired-up credentialed service. Returns [] entries only for keys that
    are actually set (an unset key isn't a failure, it's just not checked)."""
    s = get_settings()
    checks = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        if s.odds_api_key:
            checks.append(_probe(client, "odds_api_key", "The Odds API", "GET",
                                  f"{ODDS_API_BASE}/sports/", params={"apiKey": s.odds_api_key}))
        if s.api_football_key:
            checks.append(_probe(client, "api_football_key", "API-Football", "GET",
                                  f"https://{s.api_football_host}/status",
                                  headers={"x-apisports-key": s.api_football_key}))
        if s.sportmonks_api_key:
            checks.append(_probe(client, "sportmonks_api_key", "Sportmonks", "GET",
                                  "https://api.sportmonks.com/v3/football/leagues",
                                  params={"api_token": s.sportmonks_api_key, "per_page": 1}))
        if s.football_data_org_key:
            checks.append(_probe(client, "football_data_org_key", "football-data.org", "GET",
                                  "https://api.football-data.org/v4/competitions/PL",
                                  headers={"X-Auth-Token": s.football_data_org_key}))
        if s.youtube_api_key:
            checks.append(_probe(client, "youtube_api_key", "YouTube Data API", "GET",
                                  "https://www.googleapis.com/youtube/v3/i18nLanguages",
                                  params={"part": "snippet", "key": s.youtube_api_key}))
        if s.discord_webhook_url:
            checks.append(_probe(client, "discord_webhook_url", "Discord webhook", "GET",
                                  s.discord_webhook_url))
        if s.llm_api_key:
            base_url, api_key, _ = _endpoint(s.llm_default_model or "x")
            checks.append(_probe(client, "llm_api_key", "LLM (primary)", "GET",
                                  f"{base_url.rstrip('/')}/models",
                                  headers={"Authorization": f"Bearer {api_key}"}))
        if s.llm_alt_api_key:
            checks.append(_probe(client, "llm_alt_api_key", "LLM (alt)", "GET",
                                  f"{s.llm_alt_base_url.rstrip('/')}/models",
                                  headers={"Authorization": f"Bearer {s.llm_alt_api_key}"}))
        if s.ollama_enabled:
            checks.append(_probe(client, "ollama_base_url", "Ollama", "GET",
                                  f"{s.ollama_base_url.rstrip('/')}/models"))

        results = list(await asyncio.gather(*checks)) if checks else []

    if s.fpl_email and s.fpl_password:
        results.append(await _check_fpl_login())

    return results
