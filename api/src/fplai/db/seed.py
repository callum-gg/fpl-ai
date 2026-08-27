"""One-time seeds: sources, team aliases, YouTube channels, default settings."""

from __future__ import annotations

from ..defaults import (
    SEED_PLAYER_ALIASES,
    SEED_SOURCES,
    SEED_YOUTUBE_CHANNELS,
    TEAM_ALIASES,
)
from ..resolve.normalise import norm_name
from .engine import writer
from .settings_store import seed_defaults


def seed_all() -> None:
    with writer() as conn:
        for sid, name, cat, req_key, enabled, base, rate in SEED_SOURCES:
            conn.execute(
                "INSERT INTO sources(id,display_name,category,requires_key,enabled,base_url,"
                "rate_limit_per_min) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET display_name=excluded.display_name,"
                "category=excluded.category,requires_key=excluded.requires_key,"
                "base_url=excluded.base_url,rate_limit_per_min=excluded.rate_limit_per_min",
                (sid, name, cat, req_key, enabled, base, rate),
            )
        for alias, canonical in TEAM_ALIASES.items():
            conn.execute(
                "INSERT OR IGNORE INTO team_aliases(team_key,alias_norm,origin) VALUES(?,?,'seed')",
                (canonical, norm_name(alias)),
            )
        for title in SEED_YOUTUBE_CHANNELS:
            # channel_id is unknown until the YouTube connector resolves the handle;
            # placeholder rows keep the settings UI list and the DB in step.
            conn.execute(
                "INSERT OR IGNORE INTO channels(channel_id,title,tracked,added_by) "
                "VALUES(?,?,1,'seed')",
                (f"pending:{norm_name(title)}", title),
            )
    seed_defaults()


def seed_player_aliases() -> int:
    """Attach the hand-written nicknames once real players exist. Safe to re-run."""
    from .engine import query_one

    n = 0
    with writer() as conn:
        for alias, full in SEED_PLAYER_ALIASES.items():
            row = query_one(
                "SELECT id FROM players WHERE canonical_name=? OR web_name=? LIMIT 1", (full, full)
            )
            if row:
                conn.execute(
                    "INSERT OR IGNORE INTO player_aliases(player_id,alias,alias_norm,origin) "
                    "VALUES(?,?,?,'manual')",
                    (row["id"], alias, norm_name(alias)),
                )
                n += 1
    return n
