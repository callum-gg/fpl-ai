"""Squads, states, recommendations, comparison, what-if, and the FPL push. docs/09."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...config import get_settings
from ...connectors.fpl_official import next_gameweek
from ...db.engine import jdump, query, query_one, writer
from ...db.settings_store import squad_settings
from ...defaults import DEFAULT_SQUAD_SETTINGS, SQUAD_COLOURS
from ...fplsync import sync as fplsync
from ...optimiser import recommend as rec_mod
from ...rules import validate_squad

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/squads", tags=["squads"])


class SquadCreate(BaseModel):
    name: str
    colour: str | None = None
    fpl_entry_id: int | None = None
    clone_from: int | None = None
    is_shadow: bool = False


class SquadPatch(BaseModel):
    name: str | None = None
    colour: str | None = None
    fpl_entry_id: int | None = None
    settings: dict | None = None


class RecommendRequest(BaseModel):
    gameweek: int | None = None
    variants: list[str] | None = None
    force_refresh: bool = False
    use_draft: bool = False


class WhatIfRequest(BaseModel):
    constraints: dict = Field(default_factory=dict)
    gameweek: int | None = None
    variant: str = "balanced"
    use_draft: bool = False


class ManualState(BaseModel):
    gameweek: int
    bank: int = 0
    free_transfers: int = 1
    picks: list[dict]
    chips_used: list[dict] = Field(default_factory=list)


class DraftSeed(BaseModel):
    """Start (or restart) the working copy. Defaults to a copy of the set squad."""

    from_recommendation: int | None = None
    gameweek: int | None = None


class DraftEdit(BaseModel):
    """Swap players in and out. `add` ids can come from anywhere — a recommendation's
    squad, the player table, wherever the UI got them."""

    add: list[int] = Field(default_factory=list)
    drop: list[int] = Field(default_factory=list)
    captain: int | None = None
    vice: int | None = None
    bank: int | None = None


class PushExecute(BaseModel):
    recommendation_id: int
    confirmation_text: str
    preview_generated_at: str | None = None
    dry_run: bool = True


def _season() -> str:
    return get_settings().current_season


def _squad_or_404(squad_id: int) -> dict:
    row = query_one("SELECT * FROM squads WHERE id=?", (squad_id,))
    if row is None:
        raise HTTPException(404, {"error": {"code": "not_found", "message": "squad not found"}})
    return dict(row)


@router.get("")
def list_squads(include_archived: bool = False) -> list[dict]:
    rows = query(
        "SELECT * FROM squads WHERE archived=0 OR ?=1 ORDER BY created_at",
        (int(include_archived),),
    )
    gw = next_gameweek(_season())
    out = []
    for r in rows:
        d = dict(r)
        d["settings"] = squad_settings(r["id"])
        latest = query_one(
            "SELECT exp_points_gw, variant FROM recommendations WHERE squad_id=? AND gameweek=? "
            "ORDER BY generated_at DESC LIMIT 1",
            (r["id"], gw),
        )
        d["projected_points"] = latest["exp_points_gw"] if latest else None
        d.pop("settings_json", None)
        out.append(d)
    return out


@router.post("")
def create_squad(body: SquadCreate) -> dict:
    settings = dict(DEFAULT_SQUAD_SETTINGS)
    if body.clone_from:
        # Cloning copies settings — the intended workflow is "duplicate my main squad,
        # crank risk to +0.8, see what it says".
        settings = squad_settings(body.clone_from)
    n = query_one("SELECT COUNT(*) c FROM squads")["c"]
    colour = body.colour or SQUAD_COLOURS[n % len(SQUAD_COLOURS)]
    with writer() as conn:
        cur = conn.execute(
            "INSERT INTO squads(name,colour,fpl_entry_id,is_shadow,season_id,settings_json) "
            "VALUES(?,?,?,?,?,?)",
            (body.name, colour, body.fpl_entry_id, int(body.is_shadow), _season(),
             jdump(settings)),
        )
        squad_id = cur.lastrowid
        if body.clone_from:
            state = query_one(
                "SELECT * FROM squad_states WHERE squad_id=? ORDER BY captured_at DESC LIMIT 1",
                (body.clone_from,),
            )
            if state:
                rec_mod.save_state(
                    conn, squad_id, state["gameweek"], "manual",
                    [dict(r) for r in query(
                        "SELECT * FROM squad_picks WHERE squad_state_id=? ORDER BY position",
                        (state["id"],),
                    )],
                    bank=state["bank"], squad_value=state["squad_value"],
                    free_transfers=state["free_transfers"],
                    chips_used_json=state["chips_used_json"], chip_active=state["chip_active"],
                )
    return get_squad(squad_id)


@router.get("/compare")
def compare(ids: str, gw: int | None = None) -> dict:
    """Aligned rows for the side-by-side comparison view."""
    squad_ids = [int(x) for x in ids.split(",") if x.strip().isdigit()]
    gw = gw or next_gameweek(_season())
    columns = []
    all_players: list[set[int]] = []
    for sid in squad_ids:
        squad = _squad_or_404(sid)
        settings = squad_settings(sid)
        rec = query_one(
            "SELECT * FROM recommendations WHERE squad_id=? AND gameweek=? "
            "ORDER BY generated_at DESC LIMIT 1",
            (sid, gw),
        )
        payload = json.loads(rec["payload_json"]) if rec else {}
        players = {p["player_id"] for p in payload.get("squad", [])}
        all_players.append(players)
        columns.append(
            {
                "squad_id": sid,
                "name": squad["name"],
                "colour": squad["colour"],
                "risk": settings.get("risk"),
                "rank_mode": settings.get("rank_mode"),
                "horizon_gws": settings.get("horizon_gws"),
                "exp_points_gw": rec["exp_points_gw"] if rec else None,
                "exp_points_horizon": rec["exp_points_horizon"] if rec else None,
                "sd_points_gw": rec["sd_points_gw"] if rec else None,
                "hits": rec["hits_taken"] if rec else None,
                "chip": rec["chip_suggested"] if rec else None,
                "transfers": payload.get("transfers", []),
                "headline": payload.get("headline"),
                "squad": payload.get("squad", []),
                "per_gameweek": payload.get("future_gameweeks", []),
            }
        )
    shared = set.intersection(*all_players) if all_players and all(all_players) else set()
    return {
        "gameweek": gw,
        "columns": columns,
        "shared_players": sorted(shared),
        "unique_players": {
            str(c["squad_id"]): sorted(all_players[i] - shared)
            for i, c in enumerate(columns)
        },
    }


@router.get("/{squad_id}")
def get_squad(squad_id: int) -> dict:
    squad = _squad_or_404(squad_id)
    squad["settings"] = squad_settings(squad_id)
    squad.pop("settings_json", None)
    squad["state"] = rec_mod.current_state(squad_id)
    squad["leagues"] = [
        dict(r) for r in query("SELECT * FROM squad_leagues WHERE squad_id=?", (squad_id,))
    ]
    return squad


@router.patch("/{squad_id}")
def patch_squad(squad_id: int, body: SquadPatch) -> dict:
    _squad_or_404(squad_id)
    with writer() as conn:
        if body.name is not None:
            conn.execute("UPDATE squads SET name=? WHERE id=?", (body.name, squad_id))
        if body.colour is not None:
            conn.execute("UPDATE squads SET colour=? WHERE id=?", (body.colour, squad_id))
        if body.fpl_entry_id is not None:
            conn.execute("UPDATE squads SET fpl_entry_id=? WHERE id=?",
                         (body.fpl_entry_id, squad_id))
        if body.settings is not None:
            merged = {**squad_settings(squad_id), **body.settings}
            conn.execute("UPDATE squads SET settings_json=? WHERE id=?",
                         (jdump(merged), squad_id))
    return get_squad(squad_id)


@router.delete("/{squad_id}")
def archive_squad(squad_id: int) -> dict:
    _squad_or_404(squad_id)
    with writer() as conn:
        conn.execute("UPDATE squads SET archived=1 WHERE id=?", (squad_id,))
    return {"archived": squad_id}


@router.post("/{squad_id}/sync")
async def sync(squad_id: int, gameweek: int | None = None) -> dict:
    _squad_or_404(squad_id)
    try:
        return await fplsync.sync_squad(squad_id, _season(), gameweek)
    except ValueError as e:
        raise HTTPException(400, {"error": {"code": "sync_unavailable", "message": str(e)}}) from e


@router.get("/{squad_id}/state")
def get_state(squad_id: int, gameweek: int | None = None) -> dict:
    state = rec_mod.current_state(squad_id, gameweek)
    if state is None:
        raise HTTPException(404, {"error": {"code": "no_state", "message": "no squad state yet"}})
    return state


@router.put("/{squad_id}/state")
def put_state(squad_id: int, body: ManualState) -> dict:
    """Manual squad entry, for shadow squads with no linked FPL entry."""
    _squad_or_404(squad_id)
    picks = [
        {**p, "selling_price": p.get("selling_price") or p.get("purchase_price")}
        for p in body.picks
    ]
    with writer() as conn:
        rec_mod.save_state(
            conn, squad_id, body.gameweek, "manual", picks,
            bank=body.bank, squad_value=sum(p.get("purchase_price", 0) for p in body.picks),
            free_transfers=body.free_transfers, chips_used_json=jdump(body.chips_used),
        )
    return rec_mod.current_state(squad_id, body.gameweek)


# --- working copy ("draft") ------------------------------------------------------
#
# The set squad is what you actually own. A draft is a scratch copy you can rearrange
# freely — swap players, pull in a recommendation's picks, re-run safe/balanced/aggressive
# against it — without any of it counting as your real team until you commit it.


def _player_meta(player_ids: list[int]) -> dict[int, dict]:
    """position / team / current price, for validation and for pricing newly added picks."""
    if not player_ids:
        return {}
    holes = ",".join("?" * len(player_ids))
    return {
        r["player_id"]: dict(r)
        for r in query(
            "SELECT ps.player_id, ps.position, ps.team_id, p.web_name, "
            "(SELECT price FROM player_prices WHERE player_id=ps.player_id "
            " AND season_id=ps.season_id ORDER BY observed_at DESC LIMIT 1) price "
            "FROM player_seasons ps JOIN players p ON p.id=ps.player_id "
            f"WHERE ps.season_id=? AND ps.player_id IN ({holes})",
            (_season(), *player_ids),
        )
    }


def _draft_or_404(squad_id: int) -> dict:
    draft = rec_mod.draft_state(squad_id)
    if draft is None:
        raise HTTPException(
            404, {"error": {"code": "no_draft", "message": "no working copy; create one first"}}
        )
    return draft


def _save_draft(squad_id: int, gameweek: int, picks: list[dict], bank: int,
                free_transfers: int, chips_used: list, chip_active: str | None) -> dict:
    """One draft per squad: writing a new one replaces whatever was there."""
    with writer() as conn:
        conn.execute("DELETE FROM squad_states WHERE squad_id=? AND source=?",
                     (squad_id, rec_mod.DRAFT))
        rec_mod.save_state(
            conn, squad_id, gameweek, rec_mod.DRAFT, picks, bank=bank,
            squad_value=sum(p.get("purchase_price") or 0 for p in picks),
            free_transfers=free_transfers, chips_used_json=jdump(chips_used),
            chip_active=chip_active,
        )
    return _draft_with_validation(squad_id)


def _draft_with_validation(squad_id: int) -> dict:
    """The draft plus what is wrong with it. A draft is allowed to be illegal while you edit
    it — you just cannot commit it until it isn't."""
    draft = _draft_or_404(squad_id)
    meta = _player_meta([p["player_id"] for p in draft["picks"]])
    checkable = []
    for pick in draft["picks"]:
        m = meta.get(pick["player_id"])
        if m and m["team_id"]:
            checkable.append({"position": m["position"], "team_id": m["team_id"],
                              "price": pick.get("purchase_price") or m["price"] or 0})
    check = validate_squad(checkable, bank=draft["bank"])
    draft["errors"] = check.errors
    draft["ok"] = check.ok
    return draft


@router.get("/{squad_id}/draft")
def get_draft(squad_id: int) -> dict:
    _squad_or_404(squad_id)
    return _draft_with_validation(squad_id)


@router.put("/{squad_id}/draft")
def seed_draft(squad_id: int, body: DraftSeed) -> dict:
    """Copy the set squad (or a recommendation) into a fresh working copy."""
    _squad_or_404(squad_id)

    if body.from_recommendation:
        rec = query_one("SELECT * FROM recommendations WHERE id=? AND squad_id=?",
                        (body.from_recommendation, squad_id))
        if rec is None:
            raise HTTPException(
                404, {"error": {"code": "not_found", "message": "recommendation not found"}}
            )
        payload = json.loads(rec["payload_json"])
        lineup = payload.get("lineup", {})
        order = [p["player_id"] for p in lineup.get("xi", [])]
        order += [p["player_id"] for p in lineup.get("bench_order", [])]
        prices = {p["player_id"]: p.get("price") for p in payload.get("squad", [])}
        picks = [
            {"player_id": pid, "position": i, "is_captain": pid == lineup.get("captain"),
             "is_vice": pid == lineup.get("vice"),
             "purchase_price": prices.get(pid), "selling_price": prices.get(pid)}
            for i, pid in enumerate(order, start=1)
        ]
        totals = payload.get("totals", {})
        return _save_draft(squad_id, rec["gameweek"], picks, totals.get("bank_after", 0),
                           totals.get("free_transfers", 1), [], payload.get("chip"))

    state = rec_mod.current_state(squad_id)
    if state is None:
        raise HTTPException(
            400,
            {"error": {"code": "no_state", "message": "no squad set yet — set one before drafting"}},
        )
    return _save_draft(squad_id, body.gameweek or state["gameweek"], state["picks"],
                       state["bank"], state["free_transfers"],
                       json.loads(state["chips_used_json"]), state["chip_active"])


@router.patch("/{squad_id}/draft")
def edit_draft(squad_id: int, body: DraftEdit) -> dict:
    """Swap players in and out of the working copy, and set the armband."""
    _squad_or_404(squad_id)
    draft = _draft_or_404(squad_id)
    picks = {p["player_id"]: dict(p) for p in draft["picks"]}

    for pid in body.drop:
        picks.pop(pid, None)
    incoming = [pid for pid in body.add if pid not in picks]
    meta = _player_meta(incoming)
    unknown = [pid for pid in incoming if pid not in meta]
    if unknown:
        raise HTTPException(
            400,
            {"error": {"code": "unknown_player",
                       "message": f"no {_season()} record for players {unknown}"}},
        )
    for pid in incoming:
        price = meta[pid]["price"]
        picks[pid] = {"player_id": pid, "is_captain": False, "is_vice": False,
                      "purchase_price": price, "selling_price": price}

    ordered = list(picks.values())
    for i, pick in enumerate(ordered, start=1):
        pick["position"] = i
    if body.captain is not None:
        for pick in ordered:
            pick["is_captain"] = pick["player_id"] == body.captain
    if body.vice is not None:
        for pick in ordered:
            pick["is_vice"] = pick["player_id"] == body.vice

    bank = draft["bank"] if body.bank is None else body.bank
    return _save_draft(squad_id, draft["gameweek"], ordered, bank, draft["free_transfers"],
                       json.loads(draft["chips_used_json"]), draft["chip_active"])


@router.delete("/{squad_id}/draft")
def discard_draft(squad_id: int) -> dict:
    _squad_or_404(squad_id)
    with writer() as conn:
        conn.execute("DELETE FROM squad_states WHERE squad_id=? AND source=?",
                     (squad_id, rec_mod.DRAFT))
    return {"discarded": squad_id}


@router.post("/{squad_id}/draft/commit")
def commit_draft(squad_id: int) -> dict:
    """Promote the working copy to be the squad you actually own."""
    _squad_or_404(squad_id)
    draft = _draft_with_validation(squad_id)
    if not draft["ok"]:
        raise HTTPException(
            400, {"error": {"code": "invalid_squad", "message": "; ".join(draft["errors"])}}
        )
    with writer() as conn:
        rec_mod.save_state(
            conn, squad_id, draft["gameweek"], "manual", draft["picks"],
            bank=draft["bank"], squad_value=draft["squad_value"],
            free_transfers=draft["free_transfers"],
            chips_used_json=draft["chips_used_json"], chip_active=draft["chip_active"],
        )
        conn.execute("DELETE FROM squad_states WHERE squad_id=? AND source=?",
                     (squad_id, rec_mod.DRAFT))
    return rec_mod.current_state(squad_id)


@router.post("/{squad_id}/recommend")
def make_recommendation(squad_id: int, body: RecommendRequest) -> list[dict]:
    _squad_or_404(squad_id)
    gw = body.gameweek or next_gameweek(_season())
    if not body.force_refresh and not body.use_draft:
        existing = query(
            "SELECT * FROM recommendations WHERE squad_id=? AND gameweek=? "
            "AND generated_at > datetime('now','-2 hour') ORDER BY generated_at DESC",
            (squad_id, gw),
        )
        if existing:
            return [_expand(dict(r)) for r in existing]
    return [
        _expand(r)
        for r in rec_mod.recommend(squad_id, _season(), gw, body.variants,
                                   persist=not body.use_draft, use_draft=body.use_draft)
    ]


@router.get("/{squad_id}/recommendations")
def list_recommendations(squad_id: int, gw: int | None = None) -> list[dict]:
    gw = gw or next_gameweek(_season())
    rows = query(
        "SELECT * FROM recommendations WHERE squad_id=? AND gameweek=? ORDER BY generated_at DESC",
        (squad_id, gw),
    )
    return [_expand(dict(r)) for r in rows]


@router.post("/{squad_id}/whatif")
def whatif(squad_id: int, body: WhatIfRequest) -> dict:
    """Constrained re-solve: force players in or out and see the honest cost."""
    _squad_or_404(squad_id)
    gw = body.gameweek or next_gameweek(_season())
    recs = rec_mod.recommend(
        squad_id, _season(), gw, variants=[body.variant], constraints=body.constraints,
        persist=False, use_draft=body.use_draft,
    )
    if not recs:
        raise HTTPException(
            422, {"error": {"code": "infeasible",
                            "message": "no legal squad satisfies those constraints"}}
        )
    return recs[0]


@router.post("/{squad_id}/push/preview")
def push_preview(squad_id: int, recommendation_id: int) -> dict:
    _squad_or_404(squad_id)
    try:
        return fplsync.preview(squad_id, recommendation_id, _season())
    except ValueError as e:
        raise HTTPException(404, {"error": {"code": "not_found", "message": str(e)}}) from e


@router.post("/{squad_id}/push/execute")
async def push_execute(squad_id: int, body: PushExecute) -> dict:
    _squad_or_404(squad_id)
    try:
        return await fplsync.execute(
            squad_id, body.recommendation_id, _season(), body.confirmation_text,
            body.preview_generated_at, body.dry_run,
        )
    except fplsync.PushRefused as e:
        raise HTTPException(403, {"error": {"code": "push_refused", "message": str(e)}}) from e
    except ValueError as e:
        raise HTTPException(404, {"error": {"code": "not_found", "message": str(e)}}) from e


@router.get("/{squad_id}/push/history")
def push_history(squad_id: int) -> list[dict]:
    return fplsync.push_history(squad_id)


def _expand(rec: dict) -> dict:
    if isinstance(rec.get("payload_json"), str):
        rec["payload"] = json.loads(rec["payload_json"])
        rec.pop("payload_json", None)
    if isinstance(rec.get("llm_critique"), str):
        try:
            rec["llm_critique"] = json.loads(rec["llm_critique"])
        except json.JSONDecodeError:
            pass
    return rec
