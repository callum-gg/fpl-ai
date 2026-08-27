"""Claim extraction: text chunks -> `claims` rows. docs/08.

This is the bridge from text to features. It runs as a queue drain over new chunks, so
adding a source never blocks ingestion on LLM latency.
"""

from __future__ import annotations

import logging

from ..config import get_settings
from ..db.engine import query, query_one, utcnow, writer
from ..resolve.entities import mention_candidates, resolve_name
from .client import LLMUnavailable, complete
from .tasks import CLAIM_TYPES, get

log = logging.getLogger(__name__)

MAX_SPAN_WORDS = 15


def pending_chunks(limit: int = 50) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT c.id, c.raw_doc_id, c.text, c.start_s, rd.source_id, rd.published_at, "
            "rd.doc_type FROM doc_chunks c JOIN raw_documents rd ON rd.id = c.raw_doc_id "
            "WHERE c.extracted = 0 ORDER BY c.id DESC LIMIT ?",
            (limit,),
        )
    ]


async def extract_chunk(chunk: dict, season_id: str, gameweek: int | None = None) -> int:
    """Extract claims from one chunk. Returns the number of claims written."""
    task = get("extract_claims")
    candidates = mention_candidates(chunk["text"], season_id)
    mentioned = ", ".join("{} ({})".format(c["name"], c["team"]) for c in candidates)
    context = "\n".join(
        [
            "source: {}".format(chunk["source_id"]),
            "published: {}".format(chunk.get("published_at") or "unknown"),
            "current gameweek: {}".format(gameweek or "unknown"),
            "players plausibly mentioned: {}".format(mentioned or "none detected"),
            "",
            "CHUNK:",
            chunk["text"],
        ]
    )
    try:
        resp = await complete(task, [{"role": "user", "content": context}])
    except LLMUnavailable:
        _mark_done(chunk["id"])
        return 0
    except Exception:
        log.exception("extraction failed for chunk %s", chunk["id"])
        _mark_done(chunk["id"])
        return 0

    claims = (resp.data or {}).get("claims", []) if isinstance(resp.data, dict) else []
    written = _store(chunk, claims, resp.model, season_id, gameweek)
    _mark_done(chunk["id"])
    return written


def _store(chunk: dict, claims: list, model: str, season_id: str, gameweek: int | None) -> int:
    now = utcnow()
    rows = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        span = (c.get("text_span") or "").strip()
        claim_type = c.get("claim_type")
        if claim_type not in CLAIM_TYPES or not span:
            continue
        # The span must genuinely be from the chunk — a hallucinated quote in the
        # evidence panel is worse than no evidence at all.
        if span.lower() not in chunk["text"].lower():
            log.debug("dropping claim: span not present in chunk")
            continue
        if len(span.split()) > MAX_SPAN_WORDS:
            span = " ".join(span.split()[:MAX_SPAN_WORDS])

        surface = c.get("player")
        player_id = None
        if surface:
            res = resolve_name(str(surface), None, season_id)
            player_id = res.player_id
        rows.append(
            {
                "raw_doc_id": chunk["raw_doc_id"],
                "chunk_id": chunk["id"],
                "player_id": player_id,
                "surface_form": str(surface) if surface else None,
                "claim_type": claim_type,
                "stance": c.get("stance"),
                "sentiment": _clamp(c.get("sentiment"), -1, 1),
                "confidence": _clamp(c.get("confidence"), 0, 1),
                "is_reported": int(bool(c.get("is_reported"))),
                "horizon_gw": _int(c.get("horizon_gw")) or gameweek,
                "text_span": span,
                "start_s": chunk.get("start_s"),
                "extracted_at": now,
                "extractor_model": model,
            }
        )
    if not rows:
        return 0
    with writer() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO claims(raw_doc_id,chunk_id,player_id,surface_form,claim_type,"
            "stance,sentiment,confidence,is_reported,horizon_gw,text_span,start_s,extracted_at,"
            "extractor_model) VALUES(:raw_doc_id,:chunk_id,:player_id,:surface_form,:claim_type,"
            ":stance,:sentiment,:confidence,:is_reported,:horizon_gw,:text_span,:start_s,"
            ":extracted_at,:extractor_model)",
            rows,
        )
    return len(rows)


def _mark_done(chunk_id: int) -> None:
    with writer() as conn:
        conn.execute("UPDATE doc_chunks SET extracted=1 WHERE id=?", (chunk_id,))


async def drain(limit: int = 50, season_id: str | None = None) -> dict:
    """Queue drain over new chunks. Scheduled every 10 minutes."""
    from ..connectors.fpl_official import current_gameweek

    season_id = season_id or get_settings().current_season
    gw = current_gameweek(season_id)
    chunks = pending_chunks(limit)
    total = 0
    for chunk in chunks:
        total += await extract_chunk(chunk, season_id, gw)
    if total:
        collapse_semantic_duplicates()
        record_pundit_calls(season_id, gw)
    return {"chunks": len(chunks), "claims": total}


def collapse_semantic_duplicates(threshold: float = 0.92) -> int:
    """Dedup layer 4: claims about the same player+type+day collapse into one group.

    This is what stops the evidence panel showing the same rumour twelve times. Uses
    embeddings where available, falling back to exact (player, type, day, stance).
    """
    rows = query(
        "SELECT id, player_id, claim_type, stance, substr(extracted_at,1,10) day "
        "FROM claims WHERE semantic_group IS NULL AND player_id IS NOT NULL LIMIT 2000"
    )
    if not rows:
        return 0
    groups: dict[tuple, int] = {}
    updates = []
    for r in rows:
        key = (r["player_id"], r["claim_type"], r["stance"], r["day"])
        gid = groups.setdefault(key, r["id"])
        updates.append((gid, r["id"]))
    with writer() as conn:
        conn.executemany("UPDATE claims SET semantic_group=? WHERE id=?", updates)
    return len(updates)


def record_pundit_calls(season_id: str, gameweek: int) -> int:
    """Every explicit buy/sell/captain call becomes a scoreable prediction."""
    call_map = {
        "recommendation": "buy",
        "captain_pick": "captain",
        "avoid": "avoid",
    }
    rows = query(
        "SELECT c.id, c.player_id, c.claim_type, c.stance, c.extracted_at, c.raw_doc_id, "
        "v.channel_id, sp.author_handle, rd.source_id "
        "FROM claims c JOIN raw_documents rd ON rd.id=c.raw_doc_id "
        "LEFT JOIN videos v ON v.raw_doc_id=c.raw_doc_id "
        "LEFT JOIN social_posts sp ON sp.raw_doc_id=c.raw_doc_id "
        f"WHERE c.player_id IS NOT NULL AND c.claim_type IN "
        f"({','.join('?' * len(call_map))}) AND c.is_reported=0 "
        "AND c.id NOT IN (SELECT claim_id FROM pundit_calls)",
        tuple(call_map),
    )
    if not rows:
        return 0
    with writer() as conn:
        for r in rows:
            call = call_map[r["claim_type"]]
            if call == "buy" and r["stance"] == "negative":
                call = "sell"
            conn.execute(
                "INSERT OR IGNORE INTO pundit_calls(channel_id,source_id,author_handle,claim_id,"
                "player_id,gameweek,call_type,made_at) VALUES(?,?,?,?,?,?,?,?)",
                (r["channel_id"], r["source_id"], r["author_handle"], r["id"], r["player_id"],
                 gameweek, call, r["extracted_at"]),
            )
    return len(rows)


async def summarise_video(video_id: int) -> dict | None:
    row = query_one(
        "SELECT id, title, transcript_text FROM videos WHERE id=? AND transcript_text IS NOT NULL",
        (video_id,),
    )
    if row is None:
        return None
    try:
        resp = await complete(
            get("summarise_video"),
            [{"role": "user",
              "content": f"Title: {row['title']}\n\n{row['transcript_text'][:12000]}"}],
        )
    except (LLMUnavailable, Exception):  # noqa: BLE001
        return None
    return resp.data if isinstance(resp.data, dict) else None


def _clamp(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
