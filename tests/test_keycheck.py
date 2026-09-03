"""Live key verification (fplai.keycheck). docs/12: one runnable check per non-trivial
branch. No real network calls — httpx.MockTransport stands in for the external services."""

from __future__ import annotations

import asyncio

import httpx
from fplai.keycheck import _probe, verify_all


def test_probe_ok_on_2xx():
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await _probe(client, "k", "svc", "GET", "https://example.test")
    r = asyncio.run(go())
    assert r == {"key": "k", "service": "svc", "ok": True, "detail": "HTTP 200"}


def test_probe_flags_rejected_key_on_401():
    transport = httpx.MockTransport(lambda r: httpx.Response(401, json={}))
    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await _probe(client, "k", "svc", "GET", "https://example.test")
    r = asyncio.run(go())
    assert r["ok"] is False
    assert "key rejected" in r["detail"]


def test_probe_reports_connection_failure_without_raising():
    def raise_error(request):
        raise httpx.ConnectError("boom", request=request)
    transport = httpx.MockTransport(raise_error)
    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await _probe(client, "k", "svc", "GET", "https://example.test")
    r = asyncio.run(go())
    assert r["ok"] is False
    assert "ConnectError" in r["detail"]


def test_verify_all_skips_everything_when_no_keys_are_set(monkeypatch):
    from fplai import keycheck
    from fplai.config import Settings

    # Settings() reads the local .env AND OS environment variables, so on any machine
    # with credentials configured this test wasn't actually keyless and probed live
    # services. Blank every credential source before constructing it.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setattr(keycheck, "get_settings", lambda: Settings(_env_file=None))
    assert asyncio.run(verify_all()) == []


def test_relative_data_paths_anchor_on_the_repo_not_the_working_directory(monkeypatch):
    """`get_settings()` mkdirs data_dir eagerly, so a relative default meant every process
    that imported fplai from elsewhere grew its own state tree — which is how a 394 MB
    stray database ended up at api/src/data/fplai.db while the real one sat in data/."""
    from fplai.config import PROJECT_ROOT, Settings

    for var in ("DATA_DIR", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)

    s = Settings(_env_file=None)
    assert s.data_dir.is_absolute()
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.db_path.is_absolute()
    assert s.db_path == PROJECT_ROOT / "data" / "fplai.db"
    assert s.models_dir == PROJECT_ROOT / "data" / "models"

    # An absolute setting is always honoured as given.
    absolute = Settings(_env_file=None, data_dir="/tmp/elsewhere",
                        database_url="sqlite:////tmp/elsewhere/x.db")
    assert str(absolute.data_dir).replace("\\", "/").endswith("/tmp/elsewhere")
