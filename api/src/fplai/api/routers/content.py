"""Feed, search, videos, evidence, chat and the SSE event stream. docs/09."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...config import get_settings
from ...connectors.fpl_official import next_deadline
from ...db.engine import jdump, query, query_one, writer
from ...llm import chat as chat_mod
from ...llm.embed import hybrid_search
from ...optimiser.recommend import save_state

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["content"])


class ChatRequest(BaseModel):
    squad_id: int | None = None
    messages: list[dict] = Field(default_factory=list)
    session_id: int | None = None


@router.get("/feed")
def feed(
    types: str = "article,video,social,claim",
    player_id: int | None = None,
    since: str | None = None,
    limit: int = 60,
    offset: int = 0,
) -> list[dict]:
    """Everything ingested, newest first, near-duplicates collapsed."""
    wanted = {t.strip() for t in types.split(",")}
    items: list[dict] = []
    where_since = " AND published_at > ?" if since else ""
    params_since: tuple = (since,) if since else ()

    if "article" in wanted:
        rows = query(
            "SELECT a.id, a.title, a.outlet, a.url, a.published_at, a.near_dupe_group, "
            "a.raw_doc_id, substr(a.body_text,1,400) excerpt, rd.source_id "
            "FROM articles a JOIN raw_documents rd ON rd.id=a.raw_doc_id "
            f"WHERE 1=1{where_since} ORDER BY a.published_at DESC LIMIT ?",
            (*params_since, limit),
        )
        seen_groups: set[int] = set()
        for r in rows:
            d = dict(r)
            grp = d.get("near_dupe_group")
            if grp and grp in seen_groups:
                for it in items:
                    if it.get("near_dupe_group") == grp:
                        it["seen_on_sites"] = it.get("seen_on_sites", 1) + 1
                        break
                continue
            if grp:
                seen_groups.add(grp)
            d.update({"type": "article", "seen_on_sites": 1})
            items.append(d)

    if "video" in wanted:
        rows = query(
            "SELECT v.id, v.youtube_id, v.title, v.channel_title, v.published_at, v.view_count, "
            "v.raw_doc_id, v.transcript_source, v.discovered_via, ch.trust_weight "
            "FROM videos v LEFT JOIN channels ch ON ch.channel_id=v.channel_id "
            f"WHERE 1=1{where_since} ORDER BY v.published_at DESC LIMIT ?",
            (*params_since, limit),
        )
        items += [
            {**dict(r), "type": "video",
             "url": f"https://youtube.com/watch?v={r['youtube_id']}"}
            for r in rows
        ]

    if "social" in wanted:
        rows = query(
            "SELECT id, platform, author_handle, body_text, posted_at, likes, score, "
            "retrieval_method, raw_doc_id FROM social_posts "
            + ("WHERE posted_at > ? " if since else "")
            + "ORDER BY posted_at DESC LIMIT ?",
            (*params_since, limit),
        )
        items += [{**dict(r), "type": "social", "published_at": r["posted_at"]} for r in rows]

    if "claim" in wanted:
        sql = (
            "SELECT c.id, c.player_id, p.web_name player_name, c.claim_type, c.stance, "
            "c.sentiment, c.text_span, c.extracted_at published_at, c.start_s, "
            "rd.source_id, rd.url, v.youtube_id FROM claims c "
            "JOIN raw_documents rd ON rd.id=c.raw_doc_id "
            "LEFT JOIN players p ON p.id=c.player_id "
            "LEFT JOIN videos v ON v.raw_doc_id=c.raw_doc_id WHERE 1=1"
        )
        params: list = []
        if player_id:
            sql += " AND c.player_id=?"
            params.append(player_id)
        if since:
            sql += " AND c.extracted_at > ?"
            params.append(since)
        rows = query(sql + " ORDER BY c.extracted_at DESC LIMIT ?", (*params, limit))
        items += [{**dict(r), "type": "claim"} for r in rows]

    items.sort(key=lambda i: i.get("published_at") or "", reverse=True)
    return items[offset: offset + limit]


@router.get("/search")
def search(q: str, k: int = 20, source_id: str | None = None) -> list[dict]:
    return hybrid_search(q, k=k, filters={"source_id": source_id} if source_id else None)


@router.get("/videos/{video_id}")
def get_video(video_id: int) -> dict:
    row = query_one("SELECT * FROM videos WHERE id=?", (video_id,))
    if row is None:
        raise HTTPException(404, {"error": {"code": "not_found", "message": "video not found"}})
    d = dict(row)
    if d.get("transcript_json"):
        try:
            d["transcript_segments"] = json.loads(d.pop("transcript_json"))
        except json.JSONDecodeError:
            d["transcript_segments"] = []
    d["claims"] = [
        dict(r) for r in query(
            "SELECT c.*, p.web_name player_name FROM claims c "
            "LEFT JOIN players p ON p.id=c.player_id WHERE c.raw_doc_id=? "
            "ORDER BY c.start_s",
            (row["raw_doc_id"],),
        )
    ]
    for c in d["claims"]:
        if c.get("start_s") is not None:
            c["deep_link"] = (
                f"https://youtube.com/watch?v={d['youtube_id']}&t={int(c['start_s'])}s"
            )
    return d


@router.get("/recommendations/{rec_id}")
def get_recommendation(rec_id: int) -> dict:
    row = query_one("SELECT * FROM recommendations WHERE id=?", (rec_id,))
    if row is None:
        raise HTTPException(404, {"error": {"code": "not_found", "message": "not found"}})
    d = dict(row)
    d["payload"] = json.loads(d.pop("payload_json"))
    if d.get("llm_critique"):
        try:
            d["llm_critique"] = json.loads(d["llm_critique"])
        except json.JSONDecodeError:
            pass
    return d


@router.get("/recommendations/{rec_id}/evidence")
def recommendation_evidence(rec_id: int) -> list[dict]:
    return [
        dict(r)
        for r in query(
            "SELECT e.*, c.text_span, c.claim_type, c.stance, c.start_s, "
            "p.web_name player_name, rd.source_id, rd.url, v.youtube_id "
            "FROM evidence_links e LEFT JOIN claims c ON c.id=e.claim_id "
            "LEFT JOIN players p ON p.id=e.player_id "
            "LEFT JOIN raw_documents rd ON rd.id=COALESCE(e.raw_doc_id, c.raw_doc_id) "
            "LEFT JOIN videos v ON v.raw_doc_id=rd.id "
            "WHERE e.subject_type='recommendation' AND e.subject_id=?",
            (rec_id,),
        )
    ]


@router.post("/recommendations/{rec_id}/explain")
async def explain_recommendation(rec_id: int) -> dict:
    from ...llm.reason import explain

    text = await explain(rec_id)
    return {"rationale": text, "available": text is not None}


@router.post("/recommendations/{rec_id}/critique")
async def critique_recommendation(rec_id: int) -> dict:
    from ...llm.reason import critique

    result = await critique(rec_id, get_settings().current_season)
    return {"critique": result, "available": result is not None}


@router.post("/recommendations/{rec_id}/accept")
def accept(rec_id: int) -> dict:
    row = query_one("SELECT * FROM recommendations WHERE id=?", (rec_id,))
    if row is None:
        raise HTTPException(404, {"error": {"code": "not_found", "message": "not found"}})
    payload = json.loads(row["payload_json"])
    with writer() as conn:
        conn.execute(
            "UPDATE recommendations SET accepted=1, accepted_at=datetime('now') WHERE id=?",
            (rec_id,),
        )
        # Accepting writes a *planned* state. Nothing is pushed to FPL by this, and
        # `current_state` ignores this source, so it never becomes the squad you own.
        lineup = payload.get("lineup", {})
        order = [p["player_id"] for p in lineup.get("xi", [])] + [
            p["player_id"] for p in lineup.get("bench_order", [])
        ]
        prices = {p["player_id"]: p.get("price") for p in payload.get("squad", [])}
        state_id = save_state(
            conn, row["squad_id"], row["gameweek"], "planned",
            [
                {"player_id": pid, "position": i,
                 "is_captain": pid == lineup.get("captain"),
                 "is_vice": pid == lineup.get("vice"),
                 "purchase_price": prices.get(pid), "selling_price": prices.get(pid)}
                for i, pid in enumerate(order, start=1)
            ],
            bank=payload.get("totals", {}).get("bank_after", 0),
            squad_value=sum(p.get("price") or 0 for p in payload.get("squad", [])),
            free_transfers=payload.get("totals", {}).get("free_transfers", 1),
            chip_active=payload.get("chip"),
        )
    return {"accepted": rec_id, "planned_state_id": state_id}


@router.post("/recommendations/{rec_id}/reject")
def reject(rec_id: int, reason: str | None = None) -> dict:
    """Rejections feed the critique pass's memory, so it learns what you actually refuse."""
    with writer() as conn:
        conn.execute(
            "UPDATE recommendations SET accepted=0, accepted_at=datetime('now'), reject_reason=? "
            "WHERE id=?",
            (reason, rec_id),
        )
    return {"rejected": rec_id, "reason": reason}


# --- chat -------------------------------------------------------------------------


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    async def gen():
        collected = []
        async for step in chat_mod.chat(body.messages, body.squad_id):
            collected.append(step)
            yield f"data: {json.dumps(step)}\n\n"
        _save_session(body, collected)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _save_session(body: ChatRequest, steps: list[dict]) -> None:
    final = next((s for s in reversed(steps) if s.get("type") == "message"), None)
    messages = body.messages + ([{"role": "assistant", "content": final["content"]}]
                                if final else [])
    title = next((m["content"][:60] for m in body.messages if m.get("role") == "user"), "Chat")
    with writer() as conn:
        if body.session_id:
            conn.execute(
                "UPDATE chat_sessions SET messages_json=?, updated_at=datetime('now') WHERE id=?",
                (jdump(messages), body.session_id),
            )
        else:
            conn.execute(
                "INSERT INTO chat_sessions(squad_id,title,messages_json) VALUES(?,?,?)",
                (body.squad_id, title, jdump(messages)),
            )


@router.get("/chat/sessions")
def chat_sessions(limit: int = 30) -> list[dict]:
    return [
        dict(r) for r in query(
            "SELECT id, squad_id, title, created_at, updated_at FROM chat_sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
    ]


@router.get("/chat/sessions/{session_id}")
def chat_session(session_id: int) -> dict:
    row = query_one("SELECT * FROM chat_sessions WHERE id=?", (session_id,))
    if row is None:
        raise HTTPException(404, {"error": {"code": "not_found", "message": "no such session"}})
    d = dict(row)
    d["messages"] = json.loads(d.pop("messages_json", "[]"))
    return d


# --- SSE events -------------------------------------------------------------------


@router.get("/events")
async def events() -> StreamingResponse:
    """Job progress, new predictions, ingest completions and the deadline countdown.

    The frontend consumes this instead of polling.
    """
    async def gen():
        last_run_id = 0
        last_pred = ""
        while True:
            season = get_settings().current_season
            payload = {"type": "tick", "deadline": next_deadline(season)}
            yield f"data: {json.dumps(payload, default=str)}\n\n"

            run = query_one(
                "SELECT id, source_id, status, docs_new, rows_upserted FROM ingest_runs "
                "WHERE id > ? AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1",
                (last_run_id,),
            )
            if run:
                last_run_id = run["id"]
                yield f"data: {json.dumps({'type': 'ingest', **dict(run)})}\n\n"

            pred = query_one(
                "SELECT MAX(generated_at) g, COUNT(*) n FROM predictions "
                "WHERE season_id=? AND generated_at > ?",
                (season, last_pred or "1970"),
            )
            if pred and pred["g"] and pred["g"] != last_pred:
                last_pred = pred["g"]
                yield f"data: {json.dumps({'type': 'predictions', 'generated_at': last_pred})}\n\n"

            await asyncio.sleep(10)

    return StreamingResponse(gen(), media_type="text/event-stream")
