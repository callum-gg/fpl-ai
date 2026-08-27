"""Embeddings + hybrid search over chunks. docs/03 + docs/08 chat tools.

sqlite-vec when the extension loads, numpy cosine over the fallback table otherwise —
same rows either way, so search degrades in speed, never in correctness. FTS5 supplies
the lexical half of the hybrid ranking.
"""

from __future__ import annotations

import logging
import struct

import numpy as np

from ..config import get_settings
from ..db.engine import query, vec_available, writer

log = logging.getLogger(__name__)

_model = None


def _encoder():
    """sentence-transformers on GPU if EMBEDDING_DEVICE allows, else CPU, else None."""
    global _model
    if _model is not None:
        return _model
    s = get_settings()
    if s.embedding_provider != "local":
        return None
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = s.embedding_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(s.embedding_model, device=device)
        log.info("embeddings on %s using %s", device, s.embedding_model)
    except Exception as e:  # noqa: BLE001 - optional dependency
        log.info("local embeddings unavailable (%s); search falls back to FTS only", e)
        _model = None
    return _model


def encode(texts: list[str]) -> np.ndarray | None:
    model = _encoder()
    if model is None:
        return None
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _pack(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.astype(np.float32))


def _unpack(b: bytes) -> np.ndarray:
    return np.array(struct.unpack(f"{len(b) // 4}f", b), dtype=np.float32)


def embed_pending(limit: int = 200) -> int:
    rows = query(
        "SELECT id, text FROM doc_chunks WHERE embedded = 0 ORDER BY id DESC LIMIT ?", (limit,)
    )
    if not rows:
        return 0
    vectors = encode([r["text"] for r in rows])
    if vectors is None:
        # No encoder: mark them done so FTS-only search still works and the queue drains.
        with writer() as conn:
            conn.executemany(
                "UPDATE doc_chunks SET embedded=1 WHERE id=?", [(r["id"],) for r in rows]
            )
        return 0

    use_vec = vec_available()
    with writer() as conn:
        for row, vec in zip(rows, vectors, strict=False):
            blob = _pack(vec)
            if use_vec:
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_vec(chunk_id, embedding) VALUES(?,?)",
                    (row["id"], blob),
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_vec_fallback(chunk_id, embedding) VALUES(?,?)",
                    (row["id"], blob),
                )
            conn.execute("UPDATE doc_chunks SET embedded=1 WHERE id=?", (row["id"],))
            conn.execute(
                "INSERT INTO chunk_fts(rowid, text) VALUES(?,?) "
                "ON CONFLICT DO NOTHING", (row["id"], row["text"])
            )
    return len(rows)


def vector_search(q: str, k: int = 20) -> list[tuple[int, float]]:
    vec = encode([q])
    if vec is None:
        return []
    v = vec[0]
    if vec_available():
        rows = query(
            "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? "
            "ORDER BY distance LIMIT ?",
            (_pack(v), k),
        )
        return [(r["chunk_id"], 1.0 - float(r["distance"])) for r in rows]

    rows = query("SELECT chunk_id, embedding FROM chunk_vec_fallback")
    if not rows:
        return []
    ids = [r["chunk_id"] for r in rows]
    mat = np.stack([_unpack(r["embedding"]) for r in rows])
    sims = mat @ v
    top = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in top]


def fts_search(q: str, k: int = 20) -> list[tuple[int, float]]:
    safe = " ".join(w for w in q.replace('"', " ").split() if w.isalnum() or "-" in w)
    if not safe:
        return []
    try:
        rows = query(
            "SELECT rowid, bm25(chunk_fts) score FROM chunk_fts WHERE chunk_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (safe, k),
        )
    except Exception:  # noqa: BLE001 - FTS syntax errors on odd queries
        return []
    return [(r["rowid"], 1.0 / (1.0 + abs(float(r["score"])))) for r in rows]


def hybrid_search(q: str, k: int = 20, filters: dict | None = None) -> list[dict]:
    """Reciprocal rank fusion of vector and lexical results. Powers /api/search and the
    chat's `search_corpus` tool."""
    vec = vector_search(q, k * 2)
    lex = fts_search(q, k * 2)
    scores: dict[int, float] = {}
    for rank, (cid, _) in enumerate(vec):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
    for rank, (cid, _) in enumerate(lex):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (60 + rank)
    if not scores:
        return []

    top = sorted(scores.items(), key=lambda kv: -kv[1])[: k * 2]
    ids = [cid for cid, _ in top]
    placeholders = ",".join("?" * len(ids))
    rows = query(
        f"SELECT c.id, c.text, c.start_s, c.raw_doc_id, rd.source_id, rd.url, rd.published_at, "
        f"v.youtube_id, v.title video_title, a.title article_title, a.outlet "
        f"FROM doc_chunks c JOIN raw_documents rd ON rd.id=c.raw_doc_id "
        f"LEFT JOIN videos v ON v.raw_doc_id=c.raw_doc_id "
        f"LEFT JOIN articles a ON a.raw_doc_id=c.raw_doc_id "
        f"WHERE c.id IN ({placeholders})",
        tuple(ids),
    )
    by_id = {r["id"]: dict(r) for r in rows}
    out = []
    for cid, score in top:
        row = by_id.get(cid)
        if row is None:
            continue
        if filters and filters.get("source_id") and row["source_id"] != filters["source_id"]:
            continue
        row["score"] = score
        row["title"] = row.get("video_title") or row.get("article_title")
        if row.get("youtube_id") and row.get("start_s") is not None:
            # Deep link straight to the moment they said it.
            row["deep_link"] = (
                f"https://youtube.com/watch?v={row['youtube_id']}&t={int(row['start_s'])}s"
            )
        else:
            row["deep_link"] = row.get("url")
        out.append(row)
        if len(out) >= k:
            break
    return out
