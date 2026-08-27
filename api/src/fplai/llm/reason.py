"""Explanation, critique with constrained re-solve, and the weekly digest. docs/08.

The critique pass cannot mutate a plan. When it proposes an alternative, that alternative
is fed back to the optimiser as a *constrained re-solve* and shown with its real,
model-computed delta — so you see the honest cost of the LLM's idea rather than the LLM's
guess at it. That single design choice is what stops this becoming a vibes machine.
"""

from __future__ import annotations

import json
import logging

from ..db.engine import query, query_one, writer
from .client import LLMUnavailable, complete
from .tasks import get

log = logging.getLogger(__name__)


def _rec(rec_id: int) -> dict | None:
    row = query_one("SELECT * FROM recommendations WHERE id=?", (rec_id,))
    return dict(row) if row else None


def _evidence_for(rec_id: int, limit: int = 25) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT e.*, c.text_span, c.claim_type, c.stance, pl.web_name player_name "
            "FROM evidence_links e LEFT JOIN claims c ON c.id=e.claim_id "
            "LEFT JOIN players pl ON pl.id=e.player_id "
            "WHERE e.subject_type='recommendation' AND e.subject_id=? LIMIT ?",
            (rec_id, limit),
        )
    ]


def _opposing_claims(player_ids: list[int], limit: int = 15) -> list[dict]:
    if not player_ids:
        return []
    placeholders = ",".join("?" * len(player_ids))
    return [
        dict(r)
        for r in query(
            f"SELECT c.player_id, pl.web_name name, c.claim_type, c.stance, c.text_span, "
            f"c.extracted_at FROM claims c JOIN players pl ON pl.id=c.player_id "
            f"WHERE c.player_id IN ({placeholders}) AND c.stance='negative' "
            f"AND c.extracted_at > datetime('now','-10 day') "
            f"ORDER BY c.extracted_at DESC LIMIT ?",
            (*player_ids, limit),
        )
    ]


def _recent_decisions(squad_id: int, limit: int = 8) -> list[dict]:
    """The critique pass gets memory: what this manager accepted and rejected recently."""
    return [
        dict(r)
        for r in query(
            "SELECT gameweek, variant, accepted, reject_reason, exp_points_gw "
            "FROM recommendations WHERE squad_id=? AND accepted IS NOT NULL "
            "ORDER BY generated_at DESC LIMIT ?",
            (squad_id, limit),
        )
    ]


async def explain(rec_id: int, persist: bool = True) -> str | None:
    rec = _rec(rec_id)
    if rec is None:
        return None
    payload = json.loads(rec["payload_json"])
    evidence = _evidence_for(rec_id)

    content = json.dumps(
        {
            "headline": payload.get("headline"),
            "variant": rec["variant"],
            "gameweek": rec["gameweek"],
            "transfers": payload.get("transfers"),
            "hits": payload.get("hits"),
            "chip": payload.get("chip"),
            "totals": payload.get("totals"),
            "alternatives": payload.get("alternatives"),
            "delta_vs_do_nothing": payload.get("delta_vs_do_nothing"),
            "recommendation": payload.get("recommendation"),
            "chip_warnings": payload.get("chip_warnings"),
            "captain": payload.get("lineup", {}).get("captain"),
            "evidence": [
                {"player": e.get("player_name"), "type": e["evidence_type"],
                 "span": e.get("text_span"), "weight": e.get("weight")}
                for e in evidence
            ],
        },
        default=str,
    )
    try:
        resp = await complete(get("explain_recommendation"), [{"role": "user", "content": content}])
    except LLMUnavailable:
        return None
    except Exception:
        log.exception("explanation failed for rec %s", rec_id)
        return None

    if persist and resp.text:
        with writer() as conn:
            conn.execute("UPDATE recommendations SET llm_rationale=? WHERE id=?",
                         (resp.text, rec_id))
    return resp.text


async def critique(rec_id: int, season_id: str, persist: bool = True) -> dict | None:
    """Adversarial pass, then re-solve every proposed alternative for its real cost."""
    rec = _rec(rec_id)
    if rec is None:
        return None
    payload = json.loads(rec["payload_json"])
    squad_ids = [p["player_id"] for p in payload.get("squad", [])]

    content = json.dumps(
        {
            "plan": {
                "transfers": payload.get("transfers"),
                "lineup": payload.get("lineup"),
                "hits": payload.get("hits"),
                "chip": payload.get("chip"),
                "totals": payload.get("totals"),
            },
            "model_numbers": payload.get("squad"),
            "opposing_claims": _opposing_claims(squad_ids),
            "recent_decisions": _recent_decisions(rec["squad_id"]),
        },
        default=str,
    )
    try:
        resp = await complete(get("critique_recommendation"),
                              [{"role": "user", "content": content}])
    except LLMUnavailable:
        return None
    except Exception:
        log.exception("critique failed for rec %s", rec_id)
        return None

    result = resp.data if isinstance(resp.data, dict) else {}
    result["overlooked_alternatives"] = await _resolve_alternatives(
        rec, season_id, result.get("overlooked_alternatives", [])
    )

    if persist:
        with writer() as conn:
            conn.execute("UPDATE recommendations SET llm_critique=? WHERE id=?",
                         (json.dumps(result, default=str), rec_id))
    return result


async def _resolve_alternatives(rec: dict, season_id: str, alternatives: list) -> list[dict]:
    """Force each proposed swap and re-optimise the rest, reporting the true delta.

    The LLM's `est_delta` is kept alongside so the UI can show how far off its guess was —
    which over time is the most honest calibration of whether to listen to it.
    """
    from ..optimiser.recommend import recommend

    out = []
    for alt in (alternatives or [])[:3]:
        if not isinstance(alt, dict):
            continue
        swap = str(alt.get("swap", ""))
        entry = {"swap": swap, "argument": alt.get("argument"),
                 "llm_est_delta": alt.get("est_delta")}
        force_in, force_out = _parse_swap(swap, season_id)
        if force_in is None and force_out is None:
            entry["actual_delta"] = None
            entry["note"] = "could not resolve the named players"
            out.append(entry)
            continue
        try:
            resolved = recommend(
                rec["squad_id"], season_id, rec["gameweek"], variants=[rec["variant"]],
                constraints={"force_in": [force_in] if force_in else [],
                             "force_out": [force_out] if force_out else []},
                persist=False,
            )
        except Exception:  # noqa: BLE001 - an infeasible forced swap is a real answer
            resolved = []
        if resolved:
            entry["actual_delta"] = round(
                resolved[0]["exp_points_horizon"] - (rec["exp_points_horizon"] or 0), 2
            )
        else:
            entry["actual_delta"] = None
            entry["note"] = "forcing this swap makes the squad infeasible"
        out.append(entry)
    return out


def _parse_swap(swap: str, season_id: str) -> tuple[int | None, int | None]:
    from ..resolve.entities import resolve_name

    for sep in ("->", "→", " for ", " to "):
        if sep in swap:
            left, right = swap.split(sep, 1)
            out = resolve_name(left.strip(), None, season_id).player_id
            inn = resolve_name(right.strip(), None, season_id).player_id
            return inn, out
    return None, None


async def weekly_digest(squad_id: int, season_id: str, gameweek: int) -> str | None:
    rec = query_one(
        "SELECT * FROM recommendations WHERE squad_id=? AND gameweek=? AND variant='balanced' "
        "ORDER BY generated_at DESC LIMIT 1",
        (squad_id, gameweek),
    )
    if rec is None:
        return None
    payload = json.loads(rec["payload_json"])
    squad_ids = [p["player_id"] for p in payload.get("squad", [])]

    content = json.dumps(
        {
            "gameweek": gameweek,
            "headline": payload.get("headline"),
            "transfers": payload.get("transfers"),
            "totals": payload.get("totals"),
            "chip_warnings": payload.get("chip_warnings"),
            "chip_calendar": payload.get("chip_calendar", [])[:2],
            "news_on_your_players": _opposing_claims(squad_ids, limit=8),
        },
        default=str,
    )
    try:
        resp = await complete(get("weekly_digest"), [{"role": "user", "content": content}])
        return resp.text
    except (LLMUnavailable, Exception):  # noqa: BLE001
        return None


async def settings_from_prose(prose: str) -> dict | None:
    """'Make me a squad tuned for a 12-person work league where I'm 40 points behind'."""
    try:
        resp = await complete(get("settings_assistant"), [{"role": "user", "content": prose}])
    except (LLMUnavailable, Exception):  # noqa: BLE001
        return None
    return resp.data if isinstance(resp.data, dict) else None
