"""Sync must never invent an empty squad, and must use my-team when picks aren't public."""

from __future__ import annotations

import httpx
import pytest


def _squad(squad_id: int = 99, entry_id: int = 6756292) -> None:
    from fplai.db.engine import writer

    with writer() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO squads(id,season_id,name,fpl_entry_id) VALUES(?,?,?,?)",
            (squad_id, "2026-27", "t", entry_id),
        )


def _fetch(picks_status: int, my_team: dict | None = None):
    """Stand in for fetch_url: picks 404s, my-team returns whatever the test supplies."""

    async def fake(url, source_id, **kw):
        if "/picks/" in url:
            req = httpx.Request("GET", url)
            raise httpx.HTTPStatusError(
                str(picks_status), request=req, response=httpx.Response(picks_status, request=req)
            )
        if "/my-team/" in url:
            assert kw["headers"]["X-API-Authorization"] == "Bearer tok"
            return httpx.Response(200, json=my_team)
        if url.endswith("/transfers/"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"leagues": {"classic": []}, "chips": [], "current": []})

    return fake


@pytest.mark.asyncio
async def test_no_token_refuses_instead_of_writing_an_empty_state(db, monkeypatch):
    from fplai.db.engine import query
    from fplai.fplsync import sync as fs

    _squad()
    monkeypatch.setattr(fs, "fetch_url", _fetch(404))
    monkeypatch.setattr(fs, "access_token", _raise_no_token)

    with pytest.raises(ValueError, match="no public picks for GW7"):
        await fs.sync_squad(99, "2026-27", gameweek=7)

    assert not query("SELECT id FROM squad_states WHERE squad_id=99")


async def _raise_no_token() -> str:
    from fplai.fplsync.sync import NO_TOKEN, PushRefused

    raise PushRefused(NO_TOKEN)


@pytest.mark.asyncio
async def test_my_team_backfills_picks_when_they_are_not_public_yet(db, monkeypatch):
    from fplai.db.engine import query, query_one, writer
    from fplai.fplsync import sync as fs

    _squad()
    with writer() as conn:
        conn.execute("INSERT OR REPLACE INTO players(id,canonical_name) VALUES(4242,'Someone')")
        conn.execute(
            "INSERT OR REPLACE INTO player_external_ids(player_id,system,external_id,confidence,"
            "method) VALUES(4242,'fpl_element','2026-27:7',1.0,'exact')"
        )

    payload = {
        "picks": [{"element": 7, "position": 1, "purchase_price": 55, "selling_price": 53,
                   "is_captain": True, "is_vice_captain": False}],
        "chips": [{"name": "bboost", "status_for_entry": "active"}],
        "transfers": {"bank": 12, "value": 1004},
    }
    monkeypatch.setattr(fs, "fetch_url", _fetch(404, payload))

    async def token() -> str:
        return "tok"

    monkeypatch.setattr(fs, "access_token", token)

    out = await fs.sync_squad(99, "2026-27", gameweek=7)

    assert out["picks"] == 1
    assert out["bank"] == 12
    row = query_one("SELECT * FROM squad_states WHERE squad_id=99")
    assert row["squad_value"] == 1004 and row["chip_active"] == "bboost"
    # my-team states both prices, so neither may be re-derived from current price.
    pick = query("SELECT * FROM squad_picks WHERE squad_state_id=?", (row["id"],))[0]
    assert (pick["purchase_price"], pick["selling_price"]) == (55, 53)


@pytest.mark.asyncio
async def test_non_404_picks_errors_are_not_mistaken_for_a_private_squad(db, monkeypatch):
    from fplai.fplsync import sync as fs

    _squad()
    monkeypatch.setattr(fs, "fetch_url", _fetch(500))

    with pytest.raises(httpx.HTTPStatusError):
        await fs.sync_squad(99, "2026-27", gameweek=7)
