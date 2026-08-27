"""Read your real team from FPL, and (optionally, carefully) push changes back. docs/02.

Reading is free and safe. Writing is not: the login flow has changed repeatedly, has at
times required solving a bot check, stores your FPL password in `.env`, and is not
something FPL sanctions. So it is gated behind `FPL_WRITE_ENABLED=false`, requires a
typed confirmation, always dry-runs first showing the exact diff, snapshots the pre-push
state, and can never be triggered by a scheduled job — human-initiated only.
"""

from __future__ import annotations

import logging
from datetime import datetime

import httpx

from ..config import get_settings
from ..connectors.base import fetch_url
from ..db import settings_store
from ..db.engine import jdump, query, query_one, utcnow, writer
from ..resolve.entities import by_external_id
from ..rules import selling_price

log = logging.getLogger(__name__)

API = "https://fantasy.premierleague.com/api"
# FPL migrated to PingOne SSO. users.premierleague.com no longer resolves, and the hosted
# login is a DaVinci JS widget behind bot management, so there is no headless password
# login to reimplement. What still works headlessly is the refresh-token grant against
# their public SPA client, which is what everything authenticated here runs on.
TOKEN_URL = "https://account.premierleague.com/as/token"
SSO_CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"
NO_TOKEN = (
    "No FPL refresh token configured. Log in at fantasy.premierleague.com, copy the "
    "refresh_token from the browser's stored SSO session, and set FPL_REFRESH_TOKEN."
)


class PushRefused(RuntimeError):
    """Raised when a guard rejects a push. The message is shown verbatim in the UI."""


# --- auth ------------------------------------------------------------------------


async def access_token() -> str:
    """Trade the stored refresh token for a short-lived access token.

    PingOne rotates refresh tokens, so the replacement is persisted straight back to the
    settings table; losing it means pasting a fresh one out of the browser again.
    """
    stored = settings_store.get(settings_store.GLOBAL, "fpl_refresh_token")
    token = stored or get_settings().fpl_refresh_token
    if not token:
        raise PushRefused(NO_TOKEN)

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "client_id": SSO_CLIENT_ID,
                  "refresh_token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if r.status_code != 200:
        raise PushRefused(
            f"FPL token refresh failed ({r.status_code}). Refresh tokens expire and are "
            f"single-use, so paste a fresh one from the browser. {r.text[:200]}"
        )
    payload = r.json()
    rotated = payload.get("refresh_token")
    if rotated and rotated != token:
        settings_store.set_many(settings_store.GLOBAL, {"fpl_refresh_token": rotated})
    return payload["access_token"]


async def _auth_headers() -> dict[str, str]:
    return {"X-API-Authorization": f"Bearer {await access_token()}"}


async def my_team(entry_id: int) -> dict:
    """The authenticated view of a squad, readable before the deadline unlike /picks/."""
    r = await fetch_url(f"{API}/my-team/{entry_id}/", "fpl_official", headers=await _auth_headers())
    team = r.json()
    # Normalise onto the shape the public picks endpoint returns, so there is one code
    # path downstream. my-team carries real purchase/selling prices, which is strictly
    # better than reconstructing them from the transfer history.
    transfers = team.get("transfers") or {}
    active = next((c["name"] for c in team.get("chips", []) if c.get("status_for_entry") == "active"), None)
    return {
        "picks": team.get("picks", []),
        "entry_history": {"bank": transfers.get("bank"), "value": transfers.get("value")},
        "active_chip": active,
    }


# --- read -----------------------------------------------------------------------


async def sync_squad(squad_id: int, season_id: str, gameweek: int | None = None) -> dict:
    """Pull the entry's picks, bank, chips and free transfers into a squad_state."""
    squad = query_one("SELECT * FROM squads WHERE id=?", (squad_id,))
    if squad is None or not squad["fpl_entry_id"]:
        raise ValueError("squad has no linked FPL entry id")
    entry_id = squad["fpl_entry_id"]

    from ..connectors.fpl_official import current_gameweek

    gameweek = gameweek or current_gameweek(season_id)

    entry = (await fetch_url(f"{API}/entry/{entry_id}/", "fpl_official")).json()
    history = (await fetch_url(f"{API}/entry/{entry_id}/history/", "fpl_official")).json()
    try:
        picks = (await fetch_url(
            f"{API}/entry/{entry_id}/event/{gameweek}/picks/", "fpl_official"
        )).json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
        # FPL keeps a squad private until its gameweek deadline passes. The authenticated
        # my-team endpoint can still see it; without a token there is nothing to read, and
        # writing an empty state would look like a successful sync of an empty team.
        try:
            picks = await my_team(entry_id)
        except PushRefused as auth_err:
            raise ValueError(
                f"FPL has no public picks for GW{gameweek} yet — a squad only becomes "
                f"readable after that gameweek's deadline. {auth_err}"
            ) from e

    transfers = (await fetch_url(f"{API}/entry/{entry_id}/transfers/", "fpl_official")).json()
    purchase_by_element = _purchase_prices(transfers, entry_id)

    history_entry = picks.get("entry_history") or {}
    bank = history_entry.get("bank")
    bank = entry.get("last_deadline_bank") or 0 if bank is None else bank
    value = history_entry.get("value")
    value = entry.get("last_deadline_value") or 1000 if value is None else value
    chips_used = [
        {"name": c.get("name"), "gameweek": c.get("event")} for c in history.get("chips", [])
    ]
    free_transfers = _free_transfers(history, gameweek)

    prices = _current_prices(season_id)
    rows = []
    unmapped: list[int] = []
    for p in picks.get("picks", []):
        pid = by_external_id("fpl_element", f"{season_id}:{p['element']}")
        if pid is None:
            # A player FPL added since the last bootstrap. Dropping it silently would hand
            # the optimiser a squad with a phantom free slot, so say so.
            unmapped.append(p["element"])
            continue
        now_price = prices.get(pid, 40)
        # my-team states both prices outright; the public picks payload states neither.
        bought = p.get("purchase_price") or purchase_by_element.get(p["element"], now_price)
        rows.append(
            {
                "player_id": pid,
                "position": p["position"],
                "is_captain": int(bool(p.get("is_captain"))),
                "is_vice": int(bool(p.get("is_vice_captain"))),
                "purchase_price": bought,
                "selling_price": p.get("selling_price") or selling_price(bought, now_price),
            }
        )

    from ..optimiser.recommend import save_state

    with writer() as conn:
        state_id = save_state(
            conn, squad_id, gameweek, "fpl_sync", rows, bank=bank, squad_value=value,
            free_transfers=free_transfers, chips_used_json=jdump(chips_used),
            chip_active=picks.get("active_chip"),
        )

    if unmapped:
        log.warning(
            "sync squad %s GW%s: %d pick(s) have no %s player mapping (fpl elements %s); "
            "run the fpl_bootstrap job to refresh them",
            squad_id, gameweek, len(unmapped), season_id, unmapped,
        )

    await sync_leagues(squad_id, entry)
    return {"squad_state_id": state_id, "gameweek": gameweek, "picks": len(rows),
            "bank": bank, "free_transfers": free_transfers, "chips_used": chips_used,
            "unmapped_elements": unmapped}


async def sync_leagues(squad_id: int, entry: dict) -> int:
    """Mini-leagues from the entry payload. This is how squad settings get league data."""
    leagues = (entry.get("leagues") or {}).get("classic", [])
    n = 0
    with writer() as conn:
        for lg in leagues:
            if lg.get("league_type") == "s":  # skip global/sponsored leagues
                continue
            conn.execute(
                "INSERT OR IGNORE INTO squad_leagues(squad_id,league_id,league_name,league_type) "
                "VALUES(?,?,?,'classic')",
                (squad_id, lg["id"], lg.get("name")),
            )
            n += 1
    return n


async def sync_league_rivals(squad_id: int, league_id: int, max_pages: int = 2) -> int:
    """Rival entry ids from the standings, so rank-aware mode has real squads to beat."""
    ids: list[int] = []
    for page in range(1, max_pages + 1):
        data = (await fetch_url(
            f"{API}/leagues-classic/{league_id}/standings/", "fpl_official",
            params={"page_standings": page},
        )).json()
        ids += [r["entry"] for r in data.get("standings", {}).get("results", [])]
        if not data.get("standings", {}).get("has_next"):
            break
    with writer() as conn:
        conn.execute(
            "UPDATE squad_leagues SET rival_entry_ids_json=? WHERE squad_id=? AND league_id=?",
            (jdump(ids[:100]), squad_id, league_id),
        )
    return len(ids)


def _purchase_prices(transfers: list, entry_id: int) -> dict[int, int]:
    """Reconstruct what was paid for each current holding from the transfer history."""
    out: dict[int, int] = {}
    for t in sorted(transfers, key=lambda x: x.get("time", "")):
        out[t["element_in"]] = t.get("element_in_cost")
        out.pop(t["element_out"], None)
    return {k: v for k, v in out.items() if v}


def _free_transfers(history: dict, gameweek: int) -> int:
    """FPL does not expose FT directly on the public API, so derive it from transfers made."""
    current = 1
    for gw in sorted(history.get("current", []), key=lambda g: g["event"]):
        if gw["event"] >= gameweek:
            break
        made = gw.get("event_transfers", 0)
        current = min(5, max(0, current - made) + 1)
    return current


def _current_prices(season_id: str) -> dict[int, int]:
    return {
        r["player_id"]: r["price"]
        for r in query(
            "SELECT player_id, price FROM player_prices pp WHERE season_id=? AND observed_at="
            "(SELECT MAX(observed_at) FROM player_prices WHERE player_id=pp.player_id "
            " AND season_id=pp.season_id)",
            (season_id,),
        )
    }


# --- write ----------------------------------------------------------------------


async def login() -> httpx.AsyncClient:
    """A client carrying the SSO bearer token. `execute` gates on FPL_WRITE_ENABLED before
    calling this, so no write check belongs here — the read path uses it too."""
    s = get_settings()
    client = httpx.AsyncClient(
        timeout=30, follow_redirects=True,
        headers={
            "User-Agent": s.fpl_user_agent,
            "Accept": "application/json",
            **await _auth_headers(),
        },
    )
    me = await client.get(f"{API}/me/")
    if me.status_code != 200 or not (me.json() or {}).get("player"):
        await client.aclose()
        raise PushRefused(
            f"FPL rejected the SSO token (status {me.status_code}). Paste a fresh "
            "FPL_REFRESH_TOKEN from a logged-in browser session."
        )
    return client


def preview(squad_id: int, recommendation_id: int, season_id: str) -> dict:
    """The exact diff a push would make, plus every warning. Always run before executing."""
    import json

    rec = query_one("SELECT * FROM recommendations WHERE id=?", (recommendation_id,))
    if rec is None:
        raise ValueError("recommendation not found")
    payload = json.loads(rec["payload_json"])

    from ..optimiser.recommend import current_state

    state = current_state(squad_id)
    current_ids = {p["player_id"] for p in (state or {}).get("picks", [])}
    planned_ids = {p["player_id"] for p in payload.get("squad", [])}

    warnings = []
    s = get_settings()
    if not s.fpl_write_enabled:
        warnings.append("FPL_WRITE_ENABLED is false — this push will be refused.")
    if state is None:
        warnings.append("No synced squad state; sync from FPL before pushing.")
    if payload.get("hits"):
        warnings.append(f"This plan takes a -{payload['hits'] * 4} point hit.")
    if payload.get("chip"):
        warnings.append(f"This plan plays the {payload['chip']} chip and cannot be undone.")

    return {
        "recommendation_id": recommendation_id,
        "gameweek": rec["gameweek"],
        "transfers_diff": {
            "in": [p for p in payload.get("squad", []) if p["player_id"] not in current_ids],
            "out": [p for p in (state or {}).get("picks", []) if p["player_id"] not in planned_ids],
        },
        "lineup_diff": payload.get("lineup"),
        "cost": payload.get("hits", 0) * 4,
        "chip": payload.get("chip"),
        "warnings": warnings,
        "generated_at": utcnow(),
        "confirmation_required": f"PUSH GW{rec['gameweek']}",
    }


PREVIEW_TTL_S = 600


async def execute(
    squad_id: int, recommendation_id: int, season_id: str, confirmation_text: str,
    preview_generated_at: str | None = None, dry_run: bool = True,
) -> dict:
    """Push transfers then the lineup. Every guard must pass; none is optional."""
    import json

    s = get_settings()
    # The kill switch is checked before anything else touches the DB or the network, so a
    # push can never fail *past* it for an unrelated reason.
    if not s.fpl_write_enabled:
        raise PushRefused("FPL_WRITE_ENABLED is false")

    prev = preview(squad_id, recommendation_id, season_id)
    if confirmation_text != prev["confirmation_required"]:
        raise PushRefused(
            f"confirmation text must be exactly {prev['confirmation_required']!r}"
        )
    if not dry_run:
        if not preview_generated_at:
            raise PushRefused("generate a preview within the last 10 minutes before executing")
        age = (
            datetime.fromisoformat(utcnow()) - datetime.fromisoformat(preview_generated_at)
        ).total_seconds()
        if age > PREVIEW_TTL_S:
            raise PushRefused(f"preview is {age:.0f}s old; regenerate it (max {PREVIEW_TTL_S}s)")

    squad = query_one("SELECT fpl_entry_id FROM squads WHERE id=?", (squad_id,))
    entry_id = squad["fpl_entry_id"] if squad else None
    if not entry_id:
        raise PushRefused("squad has no linked FPL entry id")

    rec = query_one("SELECT * FROM recommendations WHERE id=?", (recommendation_id,))
    payload = json.loads(rec["payload_json"])
    elements = _element_map(season_id)

    transfers_body = {
        "entry": entry_id,
        "event": rec["gameweek"],
        "chip": payload.get("chip"),
        "confirmed": True,
        "transfers": [
            {
                "element_in": elements.get(t["in"]["player_id"]),
                "element_out": elements.get(t["out"]["player_id"]),
                "purchase_price": t["in"].get("price"),
                "selling_price": t["out"].get("selling_price"),
            }
            for t in payload.get("transfers", [])
        ],
    }
    lineup_body = {
        "chip": payload.get("chip"),
        "picks": _lineup_picks(payload, elements),
    }

    # Snapshot the pre-push state first, so you can always see what changed.
    pre_state = query_one(
        "SELECT id FROM squad_states WHERE squad_id=? ORDER BY captured_at DESC LIMIT 1",
        (squad_id,),
    )

    if dry_run:
        result = {"dry_run": True, "transfers_body": transfers_body, "lineup_body": lineup_body,
                  "warnings": prev["warnings"]}
        _record(squad_id, rec["gameweek"], True, True, transfers_body, "dry run",
                pre_state["id"] if pre_state else None)
        return result

    client = await login()
    try:
        responses = {}
        if transfers_body["transfers"]:
            r = await client.post(
                f"{API}/transfers/", json=transfers_body,
                headers={"Referer": "https://fantasy.premierleague.com/transfers",
                         "Content-Type": "application/json"},
            )
            responses["transfers"] = {"status": r.status_code, "body": r.text[:2000]}
            if r.status_code >= 400:
                _record(squad_id, rec["gameweek"], False, False, transfers_body, r.text[:2000],
                        pre_state["id"] if pre_state else None)
                raise PushRefused(f"FPL rejected the transfers: {r.text[:500]}")

        r = await client.post(
            f"{API}/my-team/{entry_id}/", json=lineup_body,
            headers={"Referer": "https://fantasy.premierleague.com/my-team",
                     "Content-Type": "application/json"},
        )
        responses["lineup"] = {"status": r.status_code, "body": r.text[:2000]}
        ok = r.status_code < 400
        _record(squad_id, rec["gameweek"], False, ok, {"transfers": transfers_body,
                                                       "lineup": lineup_body},
                jdump(responses), pre_state["id"] if pre_state else None)
        if not ok:
            raise PushRefused(f"FPL rejected the lineup: {r.text[:500]}")
        return {"dry_run": False, "ok": True, "responses": responses}
    finally:
        await client.aclose()


def _element_map(season_id: str) -> dict[int, int]:
    return {
        r["player_id"]: r["fpl_element_id"]
        for r in query(
            "SELECT player_id, fpl_element_id FROM player_seasons WHERE season_id=? "
            "AND fpl_element_id IS NOT NULL",
            (season_id,),
        )
    }


def _lineup_picks(payload: dict, elements: dict[int, int]) -> list[dict]:
    lineup = payload.get("lineup", {})
    out = []
    for i, p in enumerate(lineup.get("xi", []), start=1):
        out.append(
            {
                "element": elements.get(p["player_id"]),
                "position": i,
                "is_captain": p["player_id"] == lineup.get("captain"),
                "is_vice_captain": p["player_id"] == lineup.get("vice"),
            }
        )
    for i, p in enumerate(lineup.get("bench_order", []), start=12):
        out.append({"element": elements.get(p["player_id"]), "position": i,
                    "is_captain": False, "is_vice_captain": False})
    return out


def _record(squad_id, gameweek, dry_run, ok, request_body, response_text, pre_state_id) -> None:
    with writer() as conn:
        conn.execute(
            "INSERT INTO push_records(squad_id,gameweek,pushed_at,dry_run,ok,request_json,"
            "response_text,pre_state_id) VALUES(?,?,?,?,?,?,?,?)",
            (squad_id, gameweek, utcnow(), int(dry_run), int(ok), jdump(request_body),
             str(response_text)[:4000], pre_state_id),
        )


def push_history(squad_id: int, limit: int = 20) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT id, gameweek, pushed_at, dry_run, ok, response_text FROM push_records "
            "WHERE squad_id=? ORDER BY pushed_at DESC LIMIT ?",
            (squad_id, limit),
        )
    ]
