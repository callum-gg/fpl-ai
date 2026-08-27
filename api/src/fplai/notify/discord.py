"""Discord webhook notifications. docs/13 phase 10.

Silently no-ops when DISCORD_WEBHOOK_URL is unset, so nothing downstream needs to guard.
"""

from __future__ import annotations

import logging

import httpx

from ..config import get_settings
from ..db.engine import query, query_one

log = logging.getLogger(__name__)

MAX_LEN = 1900


async def notify(content: str, username: str = "FPL AI") -> bool:
    url = get_settings().discord_webhook_url
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json={"content": content[:MAX_LEN], "username": username})
            r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001 - a failed notification must never break a job
        log.warning("discord notify failed: %s", e)
        return False


async def deadline_alert(season_id: str) -> bool:
    from ..connectors.fpl_official import next_deadline

    s = get_settings()
    nxt = next_deadline(season_id)
    if not nxt:
        return False
    hours = nxt["seconds_remaining"] / 3600
    for threshold in sorted(s.notify_deadline_hours):
        if threshold - 0.5 <= hours <= threshold + 0.5:
            return await notify(
                f"⏰ **GW{nxt['gameweek']} deadline in {hours:.0f}h** "
                f"({nxt['deadline_utc']})."
            )
    return False


async def price_change_alert(season_id: str, squad_ids: list[int]) -> bool:
    """Only your holdings — an alert about players you do not own is noise."""
    if not get_settings().notify_price_changes or not squad_ids:
        return False
    placeholders = ",".join("?" * len(squad_ids))
    rows = query(
        f"SELECT p.web_name, pp.price, pp.observed_at FROM player_prices pp "
        f"JOIN players p ON p.id=pp.player_id "
        f"WHERE pp.season_id=? AND pp.observed_at > datetime('now','-1 day') "
        f"AND pp.player_id IN (SELECT sp.player_id FROM squad_picks sp "
        f"  JOIN squad_states ss ON ss.id=sp.squad_state_id "
        f"  WHERE ss.squad_id IN ({placeholders})) "
        f"ORDER BY pp.observed_at DESC",
        (season_id, *squad_ids),
    )
    changes = _diff_prices(rows)
    if not changes:
        return False
    lines = [f"{'📈' if d > 0 else '📉'} **{n}** {d / 10:+.1f}m" for n, d in changes]
    return await notify("**Price changes in your squads**\n" + "\n".join(lines))


def _diff_prices(rows) -> list[tuple[str, int]]:
    seen: dict[str, list[int]] = {}
    for r in rows:
        seen.setdefault(r["web_name"], []).append(r["price"])
    return [(name, prices[0] - prices[-1]) for name, prices in seen.items()
            if len(prices) > 1 and prices[0] != prices[-1]]


async def injury_alert(player_id: int, status: str, source: str, note: str = "") -> bool:
    if not get_settings().notify_injury_to_owned:
        return False
    owned = query_one(
        "SELECT 1 FROM squad_picks sp JOIN squad_states ss ON ss.id=sp.squad_state_id "
        "JOIN squads s ON s.id=ss.squad_id WHERE sp.player_id=? AND s.archived=0 LIMIT 1",
        (player_id,),
    )
    if not owned:
        return False
    player = query_one("SELECT web_name FROM players WHERE id=?", (player_id,))
    name = player["web_name"] if player else str(player_id)
    return await notify(f"🚑 **{name}** is now `{status}` per {source}. {note[:200]}")


async def digest(squad_id: int, season_id: str, gameweek: int) -> bool:
    from ..llm.reason import weekly_digest

    text = await weekly_digest(squad_id, season_id, gameweek)
    if not text:
        rec = query_one(
            "SELECT payload_json FROM recommendations WHERE squad_id=? AND gameweek=? "
            "ORDER BY generated_at DESC LIMIT 1",
            (squad_id, gameweek),
        )
        if not rec:
            return False
        import json

        text = json.loads(rec["payload_json"]).get("headline", "")
    squad = query_one("SELECT name FROM squads WHERE id=?", (squad_id,))
    return await notify(
        f"**{squad['name'] if squad else 'Squad'} — GW{gameweek}**\n{text}"
    )


async def chip_expiry_alert(season_id: str, gameweek: int, squad_id: int) -> bool:
    from ..optimiser.chips import expiry_warnings
    from ..optimiser.recommend import _chips_used, current_state

    warnings = expiry_warnings(gameweek, _chips_used(current_state(squad_id)))
    critical = [w for w in warnings if w["severity"] in ("critical", "high")]
    if not critical:
        return False
    return await notify(
        "⚠️ **Chip expiry**\n" + "\n".join(f"• {w['message']}" for w in critical)
    )


async def source_health_alert() -> bool:
    """A connector returning zero rows twice in a row is worth knowing about."""
    rows = query(
        "SELECT source_id, MAX(started_at) last_run, "
        "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) failures "
        "FROM ingest_runs WHERE started_at > datetime('now','-1 day') GROUP BY source_id "
        "HAVING failures >= 2"
    )
    if not rows:
        return False
    lines = [f"• `{r['source_id']}` — {r['failures']} failures since yesterday" for r in rows]
    return await notify("🔌 **Source health**\n" + "\n".join(lines))
