"""APScheduler jobs. docs/04 scheduler table.

All times UK. Cadences adapt to deadline proximity: `deadline_proximity()` returns
far|near|imminent and the turbo jobs consult it, so the app polls hard when it matters
and politely when it does not.

Nothing here ever pushes to FPL. That is human-initiated only, by design.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import asdict

from ..config import get_settings
from ..connectors import registry as connectors
from ..connectors.base import run_connector
from ..connectors.fpl_official import (
    current_gameweek,
    deadline_proximity,
    next_gameweek,
)
from ..db.engine import query, query_one, utcnow, writer
from ..db.settings_store import global_settings

log = logging.getLogger(__name__)

JOBS: dict[str, Callable] = {}


def job(name: str):
    def wrap(fn):
        JOBS[name] = fn
        return fn

    return wrap


def _season() -> str:
    return get_settings().current_season


async def run_named(name: str) -> dict:
    if name not in JOBS:
        raise KeyError(f"unknown job {name!r}; known: {', '.join(sorted(JOBS))}")
    started = utcnow()
    with writer() as conn:
        cur = conn.execute(
            "INSERT INTO job_runs(job_name,started_at,status) VALUES(?,?,'running')",
            (name, started),
        )
        run_id = cur.lastrowid
    try:
        result = JOBS[name]()
        if asyncio.iscoroutine(result):
            result = await result
        status, detail = "ok", str(result)[:2000]
    except Exception as e:
        log.exception("job %s failed", name)
        status, detail, result = "failed", str(e)[:2000], {"error": str(e)}
    with writer() as conn:
        conn.execute(
            "UPDATE job_runs SET finished_at=?, status=?, detail=? WHERE id=?",
            (utcnow(), status, detail, run_id),
        )
    return {"job": name, "status": status, "result": result}


# --- FPL core -------------------------------------------------------------------


@job("fpl_bootstrap")
async def fpl_bootstrap() -> dict:
    r = await run_connector(connectors.get("fpl_official"), _season(), {"mode": "bootstrap"})
    return asdict(r)


@job("fpl_fixtures")
async def fpl_fixtures() -> dict:
    r = await run_connector(connectors.get("fpl_official"), _season(), {"mode": "fixtures"})
    return asdict(r)


@job("fpl_element_summaries")
async def fpl_element_summaries() -> dict:
    """Daily for all; hourly for watchlist and owned players when a deadline is near."""
    season = _season()
    params: dict = {"mode": "element_summaries"}
    if deadline_proximity(season) != "far":
        params["element_ids"] = _priority_elements(season)
    r = await run_connector(connectors.get("fpl_official"), season, params)
    return asdict(r)


def _priority_elements(season: str) -> list[int]:
    watchlist = global_settings().get("watchlist", [])
    owned = [
        r["player_id"] for r in query(
            "SELECT DISTINCT sp.player_id FROM squad_picks sp JOIN squad_states ss "
            "ON ss.id=sp.squad_state_id JOIN squads s ON s.id=ss.squad_id WHERE s.archived=0"
        )
    ]
    ids = set(watchlist) | set(owned)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    return [
        r["fpl_element_id"]
        for r in query(
            f"SELECT fpl_element_id FROM player_seasons WHERE season_id=? "
            f"AND player_id IN ({placeholders}) AND fpl_element_id IS NOT NULL",
            (season, *ids),
        )
    ]


@job("fpl_entry_sync")
async def fpl_entry_sync() -> dict:
    from ..fplsync.sync import sync_squad

    season = _season()
    out = []
    for r in query("SELECT id FROM squads WHERE archived=0 AND fpl_entry_id IS NOT NULL"):
        try:
            out.append(await sync_squad(r["id"], season))
        except Exception as e:  # noqa: BLE001 - one squad failing must not stop the rest
            log.warning("entry sync failed for squad %s: %s", r["id"], e)
    return {"synced": len(out)}


@job("fpl_post_lockdown_reconcile")
async def fpl_post_lockdown_reconcile() -> dict:
    """Runs 09:30 the day after each GW's last match.

    From 2026/27, lockdown moved to 09:00 the day after the final match so post-match
    Opta review can feed BPS and DefCon. Full-time data is therefore *provisional*: we
    re-pull `event/{gw}/live`, overwrite the stats, and recompute everything derived.
    """
    season = _season()
    gw = current_gameweek(season)
    result = await run_connector(
        connectors.get("fpl_official"), season, {"mode": "live", "gameweek": gw}
    )
    await run_connector(connectors.get("fpl_official"), season, {"mode": "bootstrap"})

    row = query_one(
        "SELECT data_checked FROM gameweeks WHERE season_id=? AND gameweek=?", (season, gw)
    )
    if row and row["data_checked"]:
        from ..features.build import build_gameweek
        from ..models.backtest import resolve_pundit_calls

        build_gameweek(season, gw)
        resolve_pundit_calls(season, gw)
    return {"gameweek": gw, "reconciled": result.rows_upserted,
            "data_checked": bool(row and row["data_checked"])}


# --- other sources ----------------------------------------------------------------


def _source_job(name: str, source_id: str, params: dict | None = None):
    @job(name)
    async def _fn(_source_id=source_id, _params=params) -> dict:
        conn = connectors.get(_source_id)
        r = await run_connector(conn, _season(), _params)
        return asdict(r)

    return _fn


_source_job("odds_poll", "odds_api")
_source_job("injury_scrape_premier", "premier_injuries")
_source_job("injury_scrape_physio", "physioroom")
_source_job("news_rss", "rss_news")
_source_job("youtube_tracked", "youtube", {"mode": "tracked"})
_source_job("youtube_discovery", "youtube", {"mode": "discovery"})
_source_job("social_x", "twitter_scrape")
_source_job("bluesky", "bluesky")
_source_job("reddit", "reddit")
_source_job("understat", "understat")
_source_job("fbref", "fbref")
_source_job("sofascore_ratings", "sofascore")
_source_job("transfermarkt", "transfermarkt")
_source_job("weather", "weather")
_source_job("setpieces", "setpieces")
_source_job("euro_fixtures", "euro_fixtures")
_source_job("livefpl", "livefpl")
_source_job("lineups_poll", "api_football")


@job("injury_scrape")
async def injury_scrape() -> dict:
    results = {}
    for sid in ("premier_injuries", "physioroom"):
        r = await run_connector(connectors.get(sid), _season())
        results[sid] = r.status
    return results


@job("transcripts")
async def transcripts() -> dict:
    """Queue drain for videos lacking transcripts."""
    from ..connectors.youtube import fetch_transcript, pending_transcripts, store_transcript

    done = 0
    for v in pending_transcripts(limit=10):
        got = fetch_transcript(v["youtube_id"])
        if got:
            store_transcript(v["id"], v["raw_doc_id"], *got)
            done += 1
    return {"transcribed": done}


@job("extract_claims")
async def extract_claims() -> dict:
    from ..llm.embed import embed_pending
    from ..llm.extract import drain

    embedded = embed_pending(200)
    result = await drain(50, _season())
    return {**result, "embedded": embedded}


# --- modelling pipeline -------------------------------------------------------------


# The transfer planner scores a five-gameweek horizon (`horizon_gws`, docs/07). Both jobs
# used to do the next gameweek only, which left `horizon_points` reading zero for weeks two
# to five — the optimiser planned five weeks ahead where four of them were worth nothing.
# ponytail: a constant, not a scan of every squad's `horizon_gws`. Raise it if a squad ever
# wants to look further than the default.
PLAN_HORIZON_GWS = 5


def _horizon(season: str) -> list[int]:
    """The gameweeks the planner needs populated, skipping any past the end of the season."""
    start = next_gameweek(season)
    last = query_one("SELECT MAX(gameweek) m FROM fixtures WHERE season_id=?", (season,))
    end = min(start + PLAN_HORIZON_GWS - 1, (last and last["m"]) or start)
    return list(range(start, end + 1))


@job("build_features")
def build_features() -> dict:
    from ..features.build import build_gameweek

    season = _season()
    built = {gw: build_gameweek(season, gw) for gw in _horizon(season)}
    return {"gameweeks": list(built), "values": sum(built.values())}


@job("predict")
def predict() -> dict:
    from ..models.predict import run

    season = _season()
    runs = [run(season, gw) for gw in _horizon(season)]
    return {"gameweeks": [gw for gw in _horizon(season)],
            "players": max((r.get("players", 0) for r in runs), default=0),
            "fixtures": sum(r.get("fixtures", 0) for r in runs)}


@job("optimise_all_squads")
def optimise_all_squads() -> dict:
    from ..optimiser.recommend import recommend

    season = _season()
    gw = next_gameweek(season)
    out = []
    for r in query("SELECT id FROM squads WHERE archived=0"):
        try:
            out.append({"squad_id": r["id"], "variants": len(recommend(r["id"], season, gw))})
        except Exception as e:  # noqa: BLE001
            log.warning("optimise failed for squad %s: %s", r["id"], e)
    return {"squads": out}


@job("train_models")
def train_models() -> dict:
    from ..models.train import train_all

    return train_all()


@job("resolve_pundit_calls")
def resolve_pundit_calls_job() -> dict:
    from ..models.backtest import resolve_pundit_calls

    season = _season()
    gw = current_gameweek(season)
    return {"resolved": resolve_pundit_calls(season, gw)}


# --- notifications and housekeeping ---------------------------------------------


@job("discord_digest")
async def discord_digest() -> dict:
    from ..notify.discord import chip_expiry_alert, digest, source_health_alert

    season = _season()
    gw = next_gameweek(season)
    sent = 0
    for r in query("SELECT id FROM squads WHERE archived=0"):
        sent += int(await digest(r["id"], season, gw))
        await chip_expiry_alert(season, gw, r["id"])
    await source_health_alert()
    return {"digests_sent": sent}


@job("deadline_alerts")
async def deadline_alerts() -> dict:
    from ..notify.discord import deadline_alert, price_change_alert

    season = _season()
    squad_ids = [r["id"] for r in query("SELECT id FROM squads WHERE archived=0")]
    return {
        "deadline": await deadline_alert(season),
        "prices": await price_change_alert(season, squad_ids),
    }


@job("vacuum_analyze")
def vacuum_analyze() -> dict:
    from ..db.engine import get_conn

    conn = get_conn()
    conn.execute("ANALYZE")
    conn.execute("VACUUM")
    return {"ok": True}


# --- scheduling -----------------------------------------------------------------

# Jobs whose cadence tightens as a deadline approaches (docs/04).
TURBO = {
    "fpl_bootstrap": {"far": "0 * * * *", "near": "*/10 * * * *", "imminent": "*/5 * * * *"},
    "odds_poll": {"far": "0 */4 * * *", "near": "*/30 * * * *", "imminent": "*/15 * * * *"},
    "social_x": {"far": "*/30 * * * *", "near": "*/10 * * * *", "imminent": "*/5 * * * *"},
    "lineups_poll": {"far": "20 * * * *", "near": "*/15 * * * *", "imminent": "*/5 * * * *"},
    "predict": {"far": "0 6,12,20 * * *", "near": "0 */3 * * *", "imminent": "*/30 * * * *"},
}

DEFAULT_CADENCE = {
    "fpl_bootstrap": "0 * * * *",
    "fpl_fixtures": "5 * * * *",
    "fpl_element_summaries": "30 4 * * *",
    "fpl_entry_sync": "0 */6 * * *",
    "fpl_post_lockdown_reconcile": "30 9 * * *",
    "odds_poll": "0 */4 * * *",
    "injury_scrape": "15 */3 * * *",
    "lineups_poll": "20 * * * *",
    "news_rss": "*/20 * * * *",
    "youtube_tracked": "40 */3 * * *",
    "youtube_discovery": "0 5 * * *",
    "transcripts": "*/15 * * * *",
    "social_x": "*/30 * * * *",
    "bluesky": "*/30 * * * *",
    "reddit": "*/30 * * * *",
    "understat": "0 3 * * 2",
    "fbref": "0 4 * * 2",
    "sofascore_ratings": "0 5 * * 2",
    "transfermarkt": "0 6 * * 3",
    "weather": "0 7 * * *",
    "setpieces": "0 6 * * 3",
    "euro_fixtures": "0 5 * * 1",
    "livefpl": "0 */6 * * *",
    "extract_claims": "*/10 * * * *",
    "build_features": "*/30 * * * *",
    "predict": "0 6,12,20 * * *",
    "optimise_all_squads": "20 6,12,20 * * *",
    "train_models": "0 10 * * 2",
    "resolve_pundit_calls": "0 11 * * 2",
    "discord_digest": "0 8 * * *",
    "deadline_alerts": "0 * * * *",
    "vacuum_analyze": "0 4 1 * *",
}

_scheduler = None


def cadence_for(job_name: str) -> str:
    """Settings override, then deadline-adaptive turbo, then the default."""
    configured = global_settings().get("sources.cadence", {}).get(job_name)
    if configured:
        return configured
    if job_name in TURBO:
        try:
            return TURBO[job_name][deadline_proximity(_season())]
        except Exception:  # noqa: BLE001 - no gameweeks loaded yet
            pass
    return DEFAULT_CADENCE.get(job_name, "0 * * * *")


def scheduled_jobs() -> list[dict]:
    if _scheduler is None:
        return [{"job": n, "cron": cadence_for(n), "next_run": None} for n in sorted(DEFAULT_CADENCE)]
    return [
        {"job": j.id, "cron": str(j.trigger),
         "next_run": j.next_run_time.isoformat() if j.next_run_time else None}
        for j in _scheduler.get_jobs()
    ]


def start() -> object | None:
    """One in-process scheduler alongside the API. Set APSCHEDULER_ENABLED=false to run a
    second API container as a pure web server without duplicate jobs."""
    global _scheduler
    s = get_settings()
    if not s.apscheduler_enabled:
        log.info("APScheduler disabled")
        return None

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger

    _scheduler = AsyncIOScheduler(timezone=s.tz)
    for name in DEFAULT_CADENCE:
        if name not in JOBS:
            continue
        try:
            _scheduler.add_job(
                _wrap(name), CronTrigger.from_crontab(cadence_for(name), timezone=s.tz),
                id=name, replace_existing=True, max_instances=1, coalesce=True,
            )
        except ValueError:
            log.warning("bad cron for job %s, skipping", name)
    _scheduler.start()
    log.info("scheduler started with %d jobs", len(_scheduler.get_jobs()))
    return _scheduler


def _wrap(name: str):
    async def _run():
        await run_named(name)

    return _run


def shutdown() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
