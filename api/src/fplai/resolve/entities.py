"""Entity resolution ladder from docs/04-ingestion.md.

1. exact external id -> 2. deterministic (name, team, dob) -> 3. fuzzy (>=92 auto,
80-92 review) -> 4. LLM in-context -> 5. manual review queue.

Precision beats recall here: a wrong link poisons every text feature, so an ambiguous
bare surname resolves to None and the claim stays team-level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from ..db.engine import query, query_one, writer
from .normalise import name_variants, norm_name

log = logging.getLogger(__name__)

AUTO_ACCEPT = 92
REVIEW_FLOOR = 80


@dataclass(slots=True)
class Resolution:
    player_id: int | None
    confidence: float
    method: str  # exact|deterministic|alias|fuzzy|llm|manual|ambiguous|miss


def _team_key(name: str | None) -> str | None:
    if not name:
        return None
    row = query_one("SELECT team_key FROM team_aliases WHERE alias_norm=?", (norm_name(name),))
    return row["team_key"] if row else None


def resolve_team(surface: str | None) -> str | None:
    """Nickname -> canonical FPL short name ('Spurs', 'Man Utd')."""
    return _team_key(surface)


def by_external_id(system: str, external_id: str) -> int | None:
    row = query_one(
        "SELECT player_id FROM player_external_ids WHERE system=? AND external_id=?",
        (system, str(external_id)),
    )
    return row["player_id"] if row else None


def link_external_id(
    conn, player_id: int, system: str, external_id: str, method: str, confidence: float = 1.0
) -> None:
    conn.execute(
        "INSERT INTO player_external_ids(player_id,system,external_id,confidence,method) "
        "VALUES(?,?,?,?,?) ON CONFLICT(system,external_id) DO UPDATE SET "
        "player_id=excluded.player_id, confidence=max(confidence,excluded.confidence)",
        (player_id, system, str(external_id), confidence, method),
    )


def add_alias(conn, player_id: int, alias: str, origin: str = "derived") -> None:
    n = norm_name(alias)
    if n:
        conn.execute(
            "INSERT OR IGNORE INTO player_aliases(player_id,alias,alias_norm,origin) VALUES(?,?,?,?)",
            (player_id, alias, n, origin),
        )


def upsert_player(
    conn,
    canonical_name: str,
    first_name: str | None = None,
    last_name: str | None = None,
    web_name: str | None = None,
    birth_date: str | None = None,
    nationality: str | None = None,
) -> int:
    """Find-or-create by canonical name + birth date. Seeds aliases for every variant."""
    row = query_one(
        "SELECT id FROM players WHERE canonical_name=? AND (birth_date IS ? OR ? IS NULL)",
        (canonical_name, birth_date, birth_date),
    )
    if row:
        pid = row["id"]
        if birth_date:
            conn.execute("UPDATE players SET birth_date=? WHERE id=? AND birth_date IS NULL",
                         (birth_date, pid))
    else:
        cur = conn.execute(
            "INSERT INTO players(canonical_name,first_name,last_name,web_name,birth_date,nationality)"
            " VALUES(?,?,?,?,?,?)",
            (canonical_name, first_name, last_name, web_name, birth_date, nationality),
        )
        pid = cur.lastrowid
    for v in name_variants(first_name, last_name, web_name, canonical_name):
        add_alias(conn, pid, v, "derived")
    return pid


def _candidates(team_key: str | None, season_id: str | None) -> list[tuple[str, int]]:
    """(alias_norm, player_id) restricted to a club when we know one."""
    if team_key and season_id:
        rows = query(
            "SELECT a.alias_norm, a.player_id FROM player_aliases a "
            "JOIN player_seasons ps ON ps.player_id = a.player_id AND ps.season_id = ? "
            "JOIN teams t ON t.id = ps.team_id "
            "WHERE t.name = ? OR t.short_name = ?",
            (season_id, team_key, team_key),
        )
        if rows:
            return [(r["alias_norm"], r["player_id"]) for r in rows]
    rows = query("SELECT alias_norm, player_id FROM player_aliases")
    return [(r["alias_norm"], r["player_id"]) for r in rows]


def resolve_name(
    surface: str,
    team_hint: str | None = None,
    season_id: str | None = None,
    allow_review: bool = True,
) -> Resolution:
    """The ladder, rungs 2-3 and 5. Rung 4 (LLM) lives in llm/tasks/resolve_entity.py."""
    n = norm_name(surface)
    if not n:
        return Resolution(None, 0.0, "miss")

    team_key = _team_key(team_hint) if team_hint else None
    cands = _candidates(team_key, season_id)
    if not cands:
        return Resolution(None, 0.0, "miss")

    exact = {pid for alias, pid in cands if alias == n}
    if len(exact) == 1:
        return Resolution(exact.pop(), 1.0, "alias")
    if len(exact) > 1:
        # A bare surname shared by two players with no club context: refuse rather than guess.
        _queue_review(surface, team_hint, sorted(exact))
        return Resolution(None, 0.0, "ambiguous")

    choices = [a for a, _ in cands]
    matches = process.extract(n, choices, scorer=fuzz.token_set_ratio, limit=3)
    if not matches:
        return Resolution(None, 0.0, "miss")
    _, best_score, best_idx = matches[0]
    runner_up = matches[1][1] if len(matches) > 1 else 0
    pid = cands[best_idx][1]

    if best_score >= AUTO_ACCEPT and best_score - runner_up >= 3:
        return Resolution(pid, best_score / 100, "fuzzy")
    if best_score >= REVIEW_FLOOR:
        if allow_review:
            _queue_review(surface, team_hint, [c[1] for c in cands
                                               if c[0] in {m[0] for m in matches}])
        return Resolution(None, best_score / 100, "review")
    return Resolution(None, best_score / 100, "miss")


def _queue_review(surface: str, club: str | None, candidate_ids: list[int]) -> None:
    from ..db.engine import jdump

    with writer() as conn:
        conn.execute(
            "INSERT INTO entity_review_queue(surface_form,club_context,system,occurrences,"
            "candidates_json) VALUES(?,?,'text',1,?) "
            "ON CONFLICT(surface_form,club_context,system) DO UPDATE SET "
            "occurrences = occurrences + 1",
            (surface, club or "", jdump(candidate_ids[:10])),
        )


def review_queue(limit: int = 100) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT * FROM entity_review_queue WHERE resolved_player_id IS NULL "
            "ORDER BY occurrences DESC LIMIT ?",
            (limit,),
        )
    ]


def resolve_review_item(item_id: int, player_id: int | None) -> None:
    """Manual resolution (rung 5). Promotes the surface form to a permanent alias."""
    row = query_one("SELECT surface_form FROM entity_review_queue WHERE id=?", (item_id,))
    with writer() as conn:
        conn.execute(
            "UPDATE entity_review_queue SET resolved_player_id=?, resolved_at=datetime('now') "
            "WHERE id=?",
            (player_id, item_id),
        )
        if player_id and row:
            add_alias(conn, player_id, row["surface_form"], "manual")


def mention_candidates(text: str, season_id: str, limit: int = 40) -> list[dict]:
    """Players plausibly mentioned in a chunk — the shortlist handed to the LLM extractor."""
    n = norm_name(text)
    rows = query(
        "SELECT DISTINCT p.id, p.canonical_name, p.web_name, t.name AS team "
        "FROM players p JOIN player_seasons ps ON ps.player_id=p.id AND ps.season_id=? "
        "LEFT JOIN teams t ON t.id=ps.team_id",
        (season_id,),
    )
    out = []
    for r in rows:
        surname = norm_name(r["web_name"] or r["canonical_name"]).split(" ")[-1]
        if surname and len(surname) > 2 and surname in n:
            out.append({"player_id": r["id"], "name": r["canonical_name"], "team": r["team"]})
    return out[:limit]
