"""DB-backed settings with the precedence chain from docs/11-config.md.

.env -> settings(global) -> settings(squad) -> request-level override.
Keys whose name matches the secret pattern are never returned unredacted.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import SECRET_RE, get_settings
from ..defaults import DEFAULT_GLOBAL_SETTINGS, DEFAULT_SQUAD_SETTINGS
from .engine import jdump, query, query_one, writer

GLOBAL = "global"


def squad_scope(squad_id: int) -> str:
    return f"squad:{squad_id}"


def get(scope: str, key: str, default: Any = None) -> Any:
    row = query_one("SELECT value_json FROM settings WHERE scope=? AND key=?", (scope, key))
    if row is None:
        return default
    return json.loads(row["value_json"])


def set_many(scope: str, values: dict[str, Any]) -> None:
    with writer() as conn:
        for k, v in values.items():
            conn.execute(
                "INSERT INTO settings(scope,key,value_json,updated_at) "
                "VALUES(?,?,?,datetime('now')) "
                "ON CONFLICT(scope,key) DO UPDATE SET value_json=excluded.value_json, "
                "updated_at=excluded.updated_at",
                (scope, k, jdump(v)),
            )


def all_in_scope(scope: str) -> dict[str, Any]:
    return {r["key"]: json.loads(r["value_json"]) for r in query(
        "SELECT key, value_json FROM settings WHERE scope=?", (scope,)
    )}


def global_settings() -> dict[str, Any]:
    """Defaults merged with whatever the UI has overridden."""
    merged = json.loads(jdump(DEFAULT_GLOBAL_SETTINGS))
    merged.update(all_in_scope(GLOBAL))
    return merged


def squad_settings(squad_id: int) -> dict[str, Any]:
    merged = json.loads(jdump(DEFAULT_SQUAD_SETTINGS))
    row = query_one("SELECT settings_json FROM squads WHERE id=?", (squad_id,))
    if row and row["settings_json"]:
        merged.update(json.loads(row["settings_json"]))
    merged.update(all_in_scope(squad_scope(squad_id)))
    return merged


def source_enabled(source_id: str) -> bool:
    return bool(global_settings().get("sources.enabled", {}).get(source_id, True))


def env_view() -> list[dict]:
    """.env-derived values for the settings UI, marked read-only and redacted."""
    s = get_settings()
    out = []
    for name, value in s.redacted().items():
        out.append(
            {
                "key": name,
                "value": value,
                "read_only": True,
                "secret": bool(SECRET_RE.search(name)),
                "source": "env",
            }
        )
    return out


def seed_defaults() -> None:
    """Write defaults for any global key not already present. Idempotent."""
    existing = set(all_in_scope(GLOBAL))
    missing = {k: v for k, v in DEFAULT_GLOBAL_SETTINGS.items() if k not in existing}
    if missing:
        set_many(GLOBAL, missing)
