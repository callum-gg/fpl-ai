"""Typer CLI: backfill, ingest, train, backtest, predict, optimise. docs/01 + docs/04."""

from __future__ import annotations

import asyncio
import json
import logging
import sys

import typer

from .config import get_settings
from .db.engine import init_db, query
from .main import setup_logging

app = typer.Typer(help="FPL AI command line", no_args_is_help=True)
log = logging.getLogger(__name__)


def _utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode accented player names
    (Sánchez, Magalhães) let alone box drawing. Force UTF-8 rather than sanitising
    every echo site."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _boot() -> str:
    _utf8_stdout()
    s = get_settings()
    setup_logging(s.log_level)
    init_db()
    return s.current_season


@app.command()
def init() -> None:
    """Create the database, apply the schema and seed sources."""
    season = _boot()
    typer.echo(f"database ready at {get_settings().db_path} (season {season})")


@app.command()
def ingest(
    source: str = typer.Argument(..., help="connector id, or 'all'"),
    mode: str = typer.Option("core", help="connector-specific mode"),
    season: str | None = typer.Option(None),
) -> None:
    """Run one connector (or every available one) once."""
    season = season or _boot()
    from .connectors import registry
    from .connectors.base import run_connector

    async def _go():
        targets = registry.available() if source == "all" else [registry.get(source)]
        for c in targets:
            result = await run_connector(c, season, {"mode": mode})
            typer.echo(
                f"{c.id:22} {result.status:8} new={result.docs_new:5} "
                f"dupe={result.docs_duplicate:5} rows={result.rows_upserted:6}"
                + (f"  {result.error}" if result.error else "")
            )

    asyncio.run(_go())


@app.command()
def backfill(
    seasons: int = typer.Option(2, help="how many recent seasons of *text*; numeric takes all"),
    sources: str = typer.Option("all", help="comma-separated connector ids, or 'all'"),
) -> None:
    """The ordered backfill plan from docs/04.

    Expect 3-6 hours wall clock on a first full run, most of it sleeping on rate limits.
    """
    season = _boot()
    from .connectors import registry
    from .connectors.base import run_connector
    from .connectors.vaastav_history import SEASONS as ALL_SEASONS
    from .db.seed import seed_player_aliases

    wanted = None if sources == "all" else {s.strip() for s in sources.split(",")}

    async def _go():
        steps = [
            ("fpl_official", {"mode": "core"}, season),
            ("fpl_official", {"mode": "extras"}, season),
            # All available seasons for numeric training data — it costs nothing and the
            # minutes/goals models want every row they can get.
            ("vaastav_history", {"seasons": ALL_SEASONS}, season),
            ("understat", {"seasons": ALL_SEASONS[-4:]}, season),
            ("fbref", {"seasons": ALL_SEASONS[-2:]}, season),
            ("setpieces", {}, season),
            ("euro_fixtures", {}, season),
            ("weather", {}, season),
            # RSS only reaches back a few days. Historic text is thin and that is fine:
            # text features carry recency weight anyway.
            ("rss_news", {}, season),
            ("youtube", {"mode": "tracked"}, season),
            ("reddit", {}, season),
            ("bluesky", {}, season),
        ]
        for source_id, params, ssn in steps:
            if wanted and source_id not in wanted:
                continue
            try:
                connector = registry.get(source_id)
            except KeyError:
                continue
            typer.echo(f"-> {source_id} ...")
            result = await run_connector(connector, ssn, params)
            typer.echo(f"   {result.status}: {result.docs_new} new, {result.rows_upserted} rows")

        typer.echo("-> entity resolution pass")
        typer.echo(f"  seeded {seed_player_aliases()} manual aliases")
        from .resolve.entities import review_queue

        typer.echo(f"  {len(review_queue(10000))} surface forms awaiting manual review")
        typer.echo(
            "\nNote: there is no free historic odds source, so odds features exist only "
            "from install date forward. The model handles the missing block explicitly."
        )

    asyncio.run(_go())


@app.command("build-features")
def build_features(
    gameweek: int | None = typer.Option(None),
    season: str | None = typer.Option(None),
    all_gws: bool = typer.Option(False, "--all", help="rebuild every gameweek in the season"),
) -> None:
    """Compute and store the feature store for one gameweek or the whole season."""
    current = _boot()
    season = season or current
    from .connectors.fpl_official import next_gameweek
    from .features.build import build_gameweek

    gws = (
        [r["gameweek"] for r in query(
            "SELECT DISTINCT gameweek FROM fixtures WHERE season_id=? AND gameweek IS NOT NULL "
            "ORDER BY gameweek", (season,)
        )]
        if all_gws
        else [gameweek or next_gameweek(season)]
    )
    for gw in gws:
        n = build_gameweek(season, gw)
        typer.echo(f"GW{gw:>2}: {n} feature values")


@app.command()
def train(
    models: str = typer.Option("", help="comma-separated model names; blank = all"),
    seasons: str = typer.Option("", help="comma-separated season ids; blank = all"),
    all_models: bool = typer.Option(False, "--all"),
) -> None:
    """Walk-forward training with automatic promotion."""
    _boot()
    from .models.train import train_all

    result = train_all(
        [s.strip() for s in seasons.split(",") if s.strip()] or None,
        None if (all_models or not models) else [m.strip() for m in models.split(",")],
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def predict(
    gameweek: int | None = typer.Option(None),
    season: str | None = typer.Option(None),
    sims: int | None = typer.Option(None, help="Monte Carlo iterations"),
) -> None:
    """Run the full prediction pass for a gameweek."""
    current = _boot()
    season = season or current
    from .connectors.fpl_official import next_gameweek
    from .models.predict import run

    gw = gameweek or next_gameweek(season)
    typer.echo(json.dumps(run(season, gw, sims), indent=2, default=str))


@app.command()
def optimise(
    squad_id: int = typer.Argument(...),
    gameweek: int | None = typer.Option(None),
    variants: str = typer.Option("safe,balanced,aggressive"),
) -> None:
    """Produce recommendations for one squad."""
    season = _boot()
    from .connectors.fpl_official import next_gameweek
    from .optimiser.recommend import recommend

    gw = gameweek or next_gameweek(season)
    recs = recommend(squad_id, season, gw, [v.strip() for v in variants.split(",")])
    for r in recs:
        p = r["payload"]
        typer.echo(f"\n── {r['variant'].upper()} ──")
        typer.echo(p["headline"])
        typer.echo(
            f"  GW{gw}: {p['totals']['exp_points_gw']} pts "
            f"(±{p['totals']['sd_points_gw']}), horizon {p['totals']['exp_points_horizon']}"
        )
        for w in p.get("chip_warnings", []):
            typer.echo(f"  ! {w['message']}")


@app.command()
def backtest(
    seasons: str = typer.Option(..., help="comma-separated, e.g. 2024-25,2025-26"),
    settings: str = typer.Option("safe,balanced,aggressive"),
    start_gw: int = typer.Option(1),
    end_gw: int = typer.Option(38),
) -> None:
    """Replay seasons deadline-by-deadline and score against reality."""
    _boot()
    from .models.backtest import run_backtest

    report = run_backtest(
        [s.strip() for s in seasons.split(",")],
        [s.strip() for s in settings.split(",")],
        start_gw, end_gw,
    )
    for variant, r in report["variants"].items():
        typer.echo(
            f"{variant:12} {r['points']:5} pts over {r['gameweeks']} GWs "
            f"(avg {r['avg_gw_points']}, {r['transfers']} transfers, {r['hits']} hits)"
        )
    typer.echo("\nCaveats:")
    for c in report["caveats"]:
        typer.echo(f"  * {c}")


@app.command("create-squad")
def create_squad(
    name: str = typer.Argument(...),
    entry_id: int | None = typer.Option(None, help="your FPL entry id"),
    risk: float = typer.Option(0.0, min=-1.0, max=1.0),
) -> None:
    """Create a squad, optionally linked to a real FPL entry."""
    season = _boot()
    from .db.engine import jdump, writer
    from .defaults import DEFAULT_SQUAD_SETTINGS, SQUAD_COLOURS

    settings = {**DEFAULT_SQUAD_SETTINGS, "risk": risk}
    with writer() as conn:
        cur = conn.execute(
            "INSERT INTO squads(name,colour,fpl_entry_id,season_id,settings_json) "
            "VALUES(?,?,?,?,?)",
            (name, SQUAD_COLOURS[0], entry_id, season, jdump(settings)),
        )
    typer.echo(f"created squad {cur.lastrowid}: {name}")


@app.command()
def sync(squad_id: int = typer.Argument(...)) -> None:
    """Pull a squad's picks, bank and chips from the FPL API."""
    season = _boot()
    from .fplsync.sync import sync_squad

    typer.echo(json.dumps(asyncio.run(sync_squad(squad_id, season)), indent=2, default=str))


@app.command()
def job(name: str = typer.Argument(..., help="scheduler job name, or 'list'")) -> None:
    """Run one scheduler job by hand."""
    _boot()
    from .scheduler.jobs import JOBS, run_named

    if name == "list":
        for n in sorted(JOBS):
            typer.echo(n)
        return
    typer.echo(json.dumps(asyncio.run(run_named(name)), indent=2, default=str))


@app.command()
def sources() -> None:
    """Source health: enabled, keys present, last run."""
    _boot()
    from .connectors import registry

    for s in registry.status():
        mark = "+" if s["available"] and s["enabled"] else "·"
        last = s["last_run"]
        detail = (
            f"{last['status']} {last['started_at'][:16]} ({last['rows_upserted']} rows)"
            if last else "never run"
        )
        typer.echo(f"{mark} {s['id']:22} {s['category']:8} {detail}")
        if s["unavailable_reason"]:
            typer.echo(f"    {s['unavailable_reason']}")


@app.command("verify-keys")
def verify_keys() -> None:
    """Ping every configured API key/credential for real and report pass/fail."""
    _boot()
    from .keycheck import verify_all

    results = asyncio.run(verify_all())
    if not results:
        typer.echo("no credentialed services are configured")
        return
    for r in results:
        mark = "+" if r["ok"] else "x"
        typer.echo(f"{mark} {r['service']:20} {r['key']:24} {r['detail']}")
    failed = sum(1 for r in results if not r["ok"])
    if failed:
        typer.echo(f"\n{failed}/{len(results)} failed")
        raise typer.Exit(1)


@app.command()
def reprocess(
    source: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", help="re-parse even up-to-date documents"),
) -> None:
    """Re-parse the raw archive with the current parser. Zero refetches."""
    _boot()
    from .connectors import registry
    from .connectors.base import reprocess as _reprocess

    n = asyncio.run(_reprocess(registry.get(source), force=force))
    typer.echo(f"reprocessed {source}: {n} rows upserted")


@app.command("capture-fixtures")
def capture_fixtures(out_dir: str = typer.Option("tests/fixtures")) -> None:
    """Re-capture golden test payloads when a source's shape changes. docs/12."""
    from pathlib import Path

    _boot()
    from .connectors.base import fetch_url

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    async def _go():
        base = "https://fantasy.premierleague.com/api"
        for name, url in (
            ("bootstrap_static.json", f"{base}/bootstrap-static/"),
            ("fixtures.json", f"{base}/fixtures/"),
            ("element_summary_1.json", f"{base}/element-summary/1/"),
        ):
            try:
                r = await fetch_url(url, "fpl_official")
                (target / name).write_text(r.text, encoding="utf-8")
                typer.echo(f"captured {name} ({len(r.text)} bytes)")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"failed {name}: {e}")

    asyncio.run(_go())


@app.command()
def serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    """Run the API server."""
    import uvicorn

    s = get_settings()
    uvicorn.run("fplai.main:app", host=host or s.bind_host, port=port or s.api_port, reload=reload)


if __name__ == "__main__":
    app()
