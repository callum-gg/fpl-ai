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

    monkeypatch.setattr(keycheck, "get_settings", lambda: Settings())
    assert asyncio.run(verify_all()) == []
