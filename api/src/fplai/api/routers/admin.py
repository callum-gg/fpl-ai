"""Settings, sources, jobs, models, backtests, pundits, entity review. docs/09."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...config import get_settings
from ...connectors import registry as connectors
from ...db import settings_store
from ...db.engine import query, query_one
from ...defaults import DEFAULT_GLOBAL_SETTINGS, DEFAULT_SQUAD_SETTINGS
from ...models import backtest as backtest_mod
from ...resolve import entities

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["admin"])


class SettingsPatch(BaseModel):
    values: dict = Field(default_factory=dict)


class TrainRequest(BaseModel):
    models: list[str] | None = None
    seasons: list[str] | None = None


class BacktestRequest(BaseModel):
    seasons: list[str]
    variants: list[str] | None = None
    start_gw: int = 1
    end_gw: int = 38


class ResolveRequest(BaseModel):
    player_id: int | None = None


# --- settings -------------------------------------------------------------------


@router.get("/settings/global")
def get_global_settings() -> dict:
    return {
        "settings": settings_store.global_settings(),
        "env": settings_store.env_view(),
    }


@router.patch("/settings/global")
def patch_global_settings(body: SettingsPatch) -> dict:
    settings_store.set_many(settings_store.GLOBAL, body.values)
    return {"settings": settings_store.global_settings()}


@router.get("/settings/squad/{squad_id}")
def get_squad_settings(squad_id: int) -> dict:
    return settings_store.squad_settings(squad_id)


@router.patch("/settings/squad/{squad_id}")
def patch_squad_settings(squad_id: int, body: SettingsPatch) -> dict:
    settings_store.set_many(settings_store.squad_scope(squad_id), body.values)
    return settings_store.squad_settings(squad_id)


@router.post("/settings/verify")
async def verify_keys() -> dict:
    """Presence-check isn't enough to know a key still works. This makes one real,
    cheap call per configured credential and reports whether it actually authenticates."""
    from ...keycheck import verify_all

    results = await verify_all()
    return {"results": results, "failed": sum(1 for r in results if not r["ok"])}


@router.get("/settings/schema")
def settings_schema() -> dict:
    """The settings UI renders from this, so adding a setting needs no frontend change."""
    return {
        "global": [_field(k, v) for k, v in DEFAULT_GLOBAL_SETTINGS.items()],
        "squad": [_field(k, v) for k, v in DEFAULT_SQUAD_SETTINGS.items()],
    }


_FIELD_HINTS = {
    "risk": {"widget": "slider", "min": -1, "max": 1, "step": 0.1,
             "labels": {"-1": "Safe", "0": "Balanced", "1": "Aggressive"}},
    "horizon_gws": {"widget": "slider", "min": 1, "max": 8, "step": 1},
    "horizon_decay": {"widget": "slider", "min": 0.5, "max": 1.0, "step": 0.02},
    "bench_weight": {"widget": "slider", "min": 0.0, "max": 0.5, "step": 0.01},
    "max_hits_per_gw": {"widget": "slider", "min": 0, "max": 3, "step": 1},
    "min_expected_gain_to_act": {"widget": "slider", "min": 0, "max": 5, "step": 0.1},
    "rank_mode": {"widget": "select",
                  "options": ["maximise_points", "climb_rank", "protect_rank"]},
    "adjustment.max_points": {"widget": "slider", "min": 0, "max": 5, "step": 0.1},
    "model.odds_blend_weight": {"widget": "slider", "min": 0, "max": 1, "step": 0.05},
    "ui.theme": {"widget": "select", "options": ["dark", "light"]},
}


def _field(key: str, default) -> dict:
    kind = (
        "boolean" if isinstance(default, bool)
        else "number" if isinstance(default, (int, float))
        else "list" if isinstance(default, list)
        else "object" if isinstance(default, dict)
        else "string"
    )
    return {
        "key": key,
        "type": kind,
        "default": default,
        "group": key.split(".")[0] if "." in key else "general",
        **_FIELD_HINTS.get(key, {}),
    }


# --- sources and jobs -----------------------------------------------------------


@router.get("/sources")
def sources() -> list[dict]:
    return connectors.status()


@router.post("/sources/{source_id}/toggle")
def toggle_source(source_id: str, enabled: bool) -> dict:
    current = settings_store.global_settings().get("sources.enabled", {})
    current[source_id] = enabled
    settings_store.set_many(settings_store.GLOBAL, {"sources.enabled": current})
    return {"source_id": source_id, "enabled": enabled}


@router.get("/jobs")
def jobs() -> dict:
    from ...scheduler.jobs import JOBS, scheduled_jobs

    recent = [
        dict(r) for r in query(
            "SELECT * FROM job_runs ORDER BY started_at DESC LIMIT 50"
        )
    ]
    return {"registered": sorted(JOBS), "scheduled": scheduled_jobs(), "recent": recent}


@router.post("/jobs/{name}/run")
async def run_job(name: str) -> dict:
    from ...scheduler.jobs import run_named

    try:
        return await run_named(name)
    except KeyError as e:
        raise HTTPException(404, {"error": {"code": "unknown_job", "message": str(e)}}) from e


@router.get("/ingest-runs")
def ingest_runs(limit: int = 50) -> list[dict]:
    return [dict(r) for r in query(
        "SELECT * FROM ingest_runs ORDER BY started_at DESC LIMIT ?", (limit,)
    )]


# --- models and diagnostics ------------------------------------------------------


@router.get("/models")
def models() -> list[dict]:
    rows = query("SELECT * FROM model_versions ORDER BY model_name, trained_at DESC")
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json", "{}"))
        d["params"] = json.loads(d.pop("params_json", "{}"))
        out.append(d)
    return out


@router.post("/models/train")
async def train_models(body: TrainRequest) -> dict:
    from ...models.train import train_all

    return train_all(body.seasons, body.models)


@router.post("/models/{version_id}/promote")
def promote_model(version_id: int) -> dict:
    from ...models.base import clear_cache, promote

    promote(version_id)
    clear_cache()
    return {"promoted": version_id}


@router.get("/models/{name}/calibration")
def calibration(name: str) -> dict:
    row = query_one(
        "SELECT metrics_json FROM model_versions WHERE model_name=? AND is_active=1", (name,)
    )
    if row is None:
        raise HTTPException(404, {"error": {"code": "no_model",
                                            "message": f"no active {name} model"}})
    metrics = json.loads(row["metrics_json"])
    return {
        "model": name,
        "ece": metrics.get("calibration_ece"),
        "curve": metrics.get("calibration_curve", []),
        "importance": metrics.get("importance", {}),
        "warning": metrics.get("regime_warning"),
    }


@router.get("/models/guards")
def model_guards() -> dict:
    """docs/05 section F: no text feature may be top-3 by importance in the points model."""
    from ...models.train import text_features_not_dominant

    ok, offenders = text_features_not_dominant("goals90")
    return {
        "text_features_not_dominant": ok,
        "offenders": offenders,
        "note": "Text signal is mostly a lagging summary of stats the model already has. "
                "If it dominates, suspect leakage or a proxy, not an edge.",
    }


@router.get("/backtests")
def backtests() -> list[dict]:
    rows = query("SELECT * FROM backtest_runs ORDER BY started_at DESC LIMIT 20")
    out = []
    for r in rows:
        d = dict(r)
        d["config"] = json.loads(d.pop("config_json", "{}"))
        d["detail"] = json.loads(d.pop("detail_json", "{}"))
        out.append(d)
    return out


@router.post("/backtests")
def run_backtest(body: BacktestRequest) -> dict:
    return backtest_mod.run_backtest(body.seasons, body.variants, body.start_gw, body.end_gw)


@router.get("/ablations")
def ablations(season: str | None = None, gws: str = "1-38") -> list[dict]:
    season = season or get_settings().current_season
    start, end = (int(x) for x in gws.split("-")) if "-" in gws else (int(gws), int(gws))
    return backtest_mod.run_ablations(season, list(range(start, end + 1)))


@router.get("/pundits")
def pundits() -> dict:
    return {
        "scoreboard": backtest_mod.scoreboard(),
        "note": "Scored against a price-and-position-matched baseline. Expect this to be "
                "quietly humbling for everyone involved, including the model.",
    }


# --- entity review ---------------------------------------------------------------


@router.get("/entity-review")
def entity_review(limit: int = 100) -> list[dict]:
    items = entities.review_queue(limit)
    for item in items:
        ids = json.loads(item.get("candidates_json") or "[]")
        if ids:
            placeholders = ",".join("?" * len(ids))
            item["candidates"] = [
                dict(r) for r in query(
                    f"SELECT id, canonical_name, web_name FROM players WHERE id IN ({placeholders})",
                    tuple(ids),
                )
            ]
        else:
            item["candidates"] = []
    return items


@router.post("/entity-review/{item_id}/resolve")
def resolve_entity(item_id: int, body: ResolveRequest) -> dict:
    entities.resolve_review_item(item_id, body.player_id)
    return {"resolved": item_id, "player_id": body.player_id}


# --- LLM usage --------------------------------------------------------------------


@router.get("/llm/usage")
def llm_usage(days: int = 30) -> dict:
    from ...llm.client import available, usage_summary

    return {"available": available(), "days": days, "by_task": usage_summary(days)}
