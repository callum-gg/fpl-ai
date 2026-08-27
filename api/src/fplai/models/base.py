"""Shared model plumbing: LightGBM wrappers, calibration, artefact persistence.

Artefacts are pickled under data/models/ and registered in `model_versions`; promotion
to `is_active` is automatic only when the new version beats the incumbent on the
held-out window (docs/06 training).
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..db.engine import query_one, utcnow, writer

log = logging.getLogger(__name__)


@dataclass
class ClassifierArtifact:
    booster: object
    feature_names: list[str]
    n_classes: int
    calibrator: object | None = None
    meta: dict = field(default_factory=dict)

    def _matrix(self, features: dict[str, float | None]) -> np.ndarray:
        return np.array(
            [[_nan(features.get(f)) for f in self.feature_names]], dtype=float
        )

    def predict_proba(self, X) -> np.ndarray:
        p = self.booster.predict(np.asarray(X, dtype=float))
        p = np.asarray(p)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        return p

    def predict_one(self, features: dict[str, float | None]) -> np.ndarray:
        return self.predict_proba(self._matrix(features))[0]


@dataclass
class RegressorArtifact:
    booster: object
    feature_names: list[str]
    meta: dict = field(default_factory=dict)

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.booster.predict(np.asarray(X, dtype=float)))

    def predict_one(self, features: dict[str, float | None]) -> float:
        X = np.array([[_nan(features.get(f)) for f in self.feature_names]], dtype=float)
        return float(self.predict(X)[0])


def _nan(v) -> float:
    """LightGBM handles NaN natively, which is exactly the missing-data policy we want."""
    return float("nan") if v is None else float(v)


def to_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.array([[_nan(r.get(f)) for f in feature_names] for r in rows], dtype=float)


def fit_lgbm_multiclass(
    X, y, feature_names: list[str], n_classes: int, monotone: list[int] | None = None,
    weights=None, params: dict | None = None, num_round: int = 400,
) -> ClassifierArtifact:
    import lightgbm as lgb

    base = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_leaves": 48,
        "learning_rate": 0.05,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
    }
    base.update(params or {})
    if monotone and any(monotone):
        base["monotone_constraints"] = monotone
    ds = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y), weight=weights,
                     feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(base, ds, num_boost_round=num_round)
    return ClassifierArtifact(booster, feature_names, n_classes)


def fit_lgbm_regressor(
    X, y, feature_names: list[str], objective: str = "tweedie", weights=None,
    params: dict | None = None, num_round: int = 400,
) -> RegressorArtifact:
    import lightgbm as lgb

    base = {
        "objective": objective,
        "metric": "l2",
        "verbosity": -1,
        "num_leaves": 40,
        "learning_rate": 0.05,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
    }
    if objective == "tweedie":
        base["tweedie_variance_power"] = 1.3
    base.update(params or {})
    ds = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y, dtype=float),
                     weight=weights, feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(base, ds, num_boost_round=num_round)
    return RegressorArtifact(booster, feature_names)


def fit_lgbm_binary(
    X, y, feature_names: list[str], weights=None, params: dict | None = None, num_round: int = 400
) -> ClassifierArtifact:
    import lightgbm as lgb

    base = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
            "num_leaves": 40, "learning_rate": 0.05, "min_data_in_leaf": 50}
    base.update(params or {})
    ds = lgb.Dataset(np.asarray(X, dtype=float), label=np.asarray(y), weight=weights,
                     feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(base, ds, num_boost_round=num_round)
    return ClassifierArtifact(booster, feature_names, 2)


# --- metrics --------------------------------------------------------------------


def ece(y_true, y_prob, bins: int = 10) -> float:
    """Expected calibration error. Reported for every probability the app shows."""
    y_true, y_prob = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (y_prob >= edges[i]) & (y_prob < edges[i + 1] if i < bins - 1 else y_prob <= 1)
        if not m.any():
            continue
        total += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(total)


def expected_calibration_curve(y_true, y_prob, bins: int = 10) -> list[dict]:
    y_true, y_prob = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (y_prob >= edges[i]) & (y_prob < edges[i + 1] if i < bins - 1 else y_prob <= 1)
        if not m.any():
            continue
        out.append(
            {"bin": float((edges[i] + edges[i + 1]) / 2),
             "predicted": float(y_prob[m].mean()),
             "actual": float(y_true[m].mean()),
             "n": int(m.sum())}
        )
    return out


def log_loss(y_true, y_prob) -> float:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-9, 1 - 1e-9)
    y_true = np.asarray(y_true)
    if y_prob.ndim == 1:
        return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))
    return float(-np.mean(np.log(y_prob[np.arange(len(y_true)), y_true])))


def spearman(a, b) -> float:
    """Rank correlation. This matters more than MAE — you only ever act on the ranking."""
    from scipy.stats import spearmanr

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return 0.0
    r = spearmanr(a[mask], b[mask]).statistic
    return 0.0 if math.isnan(r) else float(r)


def mae(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0


def rmse(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else 0.0


def gain_importance(artifact) -> dict[str, float]:
    try:
        gains = artifact.booster.feature_importance(importance_type="gain")
        return dict(zip(artifact.feature_names, [float(g) for g in gains], strict=False))
    except Exception:  # noqa: BLE001 - heuristic fallbacks have no booster
        return {}


# --- persistence ----------------------------------------------------------------


def save_artifact(model_name: str, artifact, metrics: dict, params: dict,
                  train_rows: int, train_seasons: list[str], feature_version: int) -> int:
    settings = get_settings()
    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = Path(settings.models_dir) / f"{model_name}-{version}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(artifact, fh)

    with writer() as conn:
        cur = conn.execute(
            "INSERT INTO model_versions(model_name,version,trained_at,train_rows,train_seasons,"
            "metrics_json,params_json,artefact_path,feature_version,is_active) "
            "VALUES(?,?,?,?,?,?,?,?,?,0)",
            (model_name, version, utcnow(), train_rows,
             ",".join(train_seasons), json.dumps(metrics), json.dumps(params, default=str),
             str(path), feature_version),
        )
        model_id = cur.lastrowid

    if _should_promote(model_name, metrics):
        promote(model_id)
    else:
        log.warning("model %s v%s not promoted: it did not beat the incumbent", model_name, version)
    return model_id


# Metric -> whether higher is better. Promotion compares only on the primary metric.
PRIMARY_METRIC = {
    "minutes": ("log_loss", False),
    "team_goals": ("log_loss", False),
    "goals90": ("spearman", True),
    "assists90": ("spearman", True),
    "defcon": ("log_loss", False),
    "bonus": ("mae", False),
    "saves90": ("mae", False),
    "cards90": ("mae", False),
    "price": ("log_loss", False),
}


# Metrics whose presence marks a scoring-definition change. If the new version has one
# and the incumbent does not, their primary metrics were not produced the same way.
METRIC_DEFINITION_MARKERS = {"team_goals": ("baseline_nll",)}


def _should_promote(model_name: str, metrics: dict) -> bool:
    if not get_settings().model_auto_promote:
        return False
    key, higher_better = PRIMARY_METRIC.get(model_name, ("log_loss", False))
    new = metrics.get(key)
    if new is None:
        return True
    incumbent = query_one(
        "SELECT metrics_json FROM model_versions WHERE model_name=? AND is_active=1", (model_name,)
    )
    if incumbent is None:
        return True
    incumbent_metrics = json.loads(incumbent["metrics_json"])
    old = incumbent_metrics.get(key)
    if old is None:
        return True
    # A score is only comparable to one computed the same way. `team_goals` used to be
    # scored in-sample, which flattered it by ~0.11 nats — enough that no honestly scored
    # successor could ever beat it, freezing the incumbent permanently. `baseline_nll`
    # only exists on versions scored against a real holdout, so its absence dates the
    # incumbent to the old definition and the comparison is void.
    for marker in METRIC_DEFINITION_MARKERS.get(model_name, ()):
        if marker in metrics and marker not in incumbent_metrics:
            log.info("%s: incumbent predates the current %s definition, promoting on that",
                     model_name, key)
            return True
    return new > old if higher_better else new < old


def promote(model_version_id: int) -> None:
    row = query_one("SELECT model_name FROM model_versions WHERE id=?", (model_version_id,))
    if row is None:
        return
    with writer() as conn:
        conn.execute("UPDATE model_versions SET is_active=0 WHERE model_name=?", (row["model_name"],))
        conn.execute("UPDATE model_versions SET is_active=1 WHERE id=?", (model_version_id,))
    log.info("promoted %s version id %s", row["model_name"], model_version_id)


_cache: dict[str, object] = {}


def load_active(model_name: str):
    """Active artefact, memoised. Returns None if nothing has been trained yet — every
    caller degrades to a heuristic so the app works before the first training run."""
    row = query_one(
        "SELECT id, artefact_path FROM model_versions WHERE model_name=? AND is_active=1 "
        "ORDER BY trained_at DESC LIMIT 1",
        (model_name,),
    )
    if row is None:
        return None
    key = f"{model_name}:{row['id']}"
    if key not in _cache:
        try:
            with open(row["artefact_path"], "rb") as fh:
                _cache[key] = pickle.load(fh)
        except (OSError, pickle.UnpicklingError) as e:
            log.warning("could not load %s: %s", model_name, e)
            return None
    return _cache[key]


def clear_cache() -> None:
    _cache.clear()


def season_weight(season_id: str, current_season: str, decay: float = 0.72) -> float:
    """w = decay ^ seasons_ago. Enough history to fit, recent enough to matter."""
    try:
        ago = int(current_season.split("-")[0]) - int(season_id.split("-")[0])
    except (ValueError, IndexError):
        return 1.0
    return float(decay ** max(0, ago))
