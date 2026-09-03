"""Connector contract + the framework that surrounds it. See docs/04-ingestion.md.

A connector only fetches and parses. Everything cross-cutting — hashing, dedup,
archiving, rate limiting, retries, circuit breaking, run accounting — lives here, so
each source file stays small and testable.

The key property: `parse` is a pure function of a RawDoc. Bump `parser_version` and the
framework reprocesses the whole archive with zero refetches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import httpx

from ..config import Settings, get_settings
from ..db.engine import jdump, query, query_one, utcnow, writer

log = logging.getLogger(__name__)

# Fields that change on every fetch without the content meaning anything different.
VOLATILE_KEYS = {
    "now", "timestamp", "fetched_at", "request_id", "requestId", "server_time",
    "last_updated", "lastUpdated", "generated_at", "etag", "cache_key", "ts",
}

_WS = re.compile(r"\s+")


def strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def content_hash(payload: Any) -> str:
    """SHA-256 over *normalised* content (docs/03, dedup layer 2)."""
    if isinstance(payload, (dict, list)):
        norm = json.dumps(strip_volatile(payload), sort_keys=True, separators=(",", ":"))
    else:
        norm = _WS.sub(" ", str(payload)).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# Calibrated empirically: reworded syndications land at 8-11 bits, unrelated football
# stories at 25+. 14 sits in that gap with margin on both sides. Collapsing two genuinely
# different stories loses information, so the threshold leans toward the cautious side.
NEAR_DUPE_DISTANCE = 14


def simhash64(text: str) -> int:
    """64-bit SimHash for near-duplicate wire copy (dedup layer 3).

    Features are unigrams plus bigrams, weighted by frequency. Unigrams carry the bulk
    of the signal and survive the insertions and deletions a subeditor makes; bigrams add
    just enough word-order sensitivity to separate two stories that share a vocabulary.
    Pure n-gram shingling was tried first and is too brittle on short copy — reordering a
    single clause moves far too many bits.
    """
    words = [w for w in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split(" ") if w]
    if not words:
        return 0
    features: dict[str, int] = {}
    for w in words:
        features[w] = features.get(w, 0) + 2                      # unigrams, weight 2
    for a, b in zip(words, words[1:], strict=False):
        key = f"{a}_{b}"
        features[key] = features.get(key, 0) + 1                  # bigrams, weight 1

    v = [0] * 64
    for feature, weight in features.items():
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for b in range(64):
            v[b] += weight if (h >> b) & 1 else -weight
    out = 0
    for b in range(64):
        if v[b] > 0:
            out |= 1 << b
    return out - (1 << 64) if out >= (1 << 63) else out  # keep it in SQLite's signed range


def hamming(a: int, b: int) -> int:
    return ((a ^ b) & ((1 << 64) - 1)).bit_count()


@dataclass(slots=True)
class RawDoc:
    doc_type: str
    payload: Any
    external_id: str | None = None
    url: str | None = None
    published_at: str | None = None
    meta: dict = field(default_factory=dict)
    text_for_simhash: str | None = None


@dataclass(slots=True)
class ParsedBatch:
    """Typed rows the framework upserts. `table -> (rows, conflict_keys)`."""

    tables: dict[str, tuple[list[dict], list[str]]] = field(default_factory=dict)
    # Work needing the live DB (entity resolution can't be expressed as a plain row).
    # Each runs inside the same writer transaction and returns a row count.
    deferred: list[Callable[[Any], int]] = field(default_factory=list)
    note: str | None = None

    def add(self, table: str, rows: list[dict], keys: list[str]) -> None:
        if not rows:
            return
        existing, _ = self.tables.get(table, ([], keys))
        self.tables[table] = (existing + rows, keys)

    def defer(self, fn: Callable[[Any], int]) -> None:
        self.deferred.append(fn)

    @property
    def row_count(self) -> int:
        return sum(len(r) for r, _ in self.tables.values())


@dataclass(slots=True)
class IngestContext:
    settings: Settings
    run_id: int
    season_id: str
    params: dict = field(default_factory=dict)
    client: httpx.AsyncClient | None = None


@dataclass(slots=True)
class IngestResult:
    source_id: str
    docs_new: int = 0
    docs_duplicate: int = 0
    rows_upserted: int = 0
    requests_made: int = 0
    status: str = "ok"
    error: str | None = None


class RateLimiter:
    """Per-source token bucket with a minimum gap (FBref needs 3s+ or it 403s you)."""

    def __init__(self, per_min: float, min_gap_s: float = 0.0):
        self.interval = 60.0 / max(per_min, 0.1)
        self.min_gap = min_gap_s
        self._next_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            if wait:
                await asyncio.sleep(wait)
            gap = max(self.interval, self.min_gap)
            self._next_at = max(now, self._next_at) + gap + random.uniform(0, gap * 0.15)


_limiters: dict[str, RateLimiter] = {}


def limiter_for(source_id: str, per_min: float = 60, min_gap_s: float = 0.0) -> RateLimiter:
    if source_id not in _limiters:
        _limiters[source_id] = RateLimiter(per_min, min_gap_s)
    return _limiters[source_id]


class CircuitOpen(RuntimeError):
    pass


_failures: dict[str, int] = {}
FAILURE_LIMIT = 5


def record_failure(source_id: str) -> int:
    _failures[source_id] = _failures.get(source_id, 0) + 1
    return _failures[source_id]


def record_success(source_id: str) -> None:
    _failures.pop(source_id, None)


def circuit_open(source_id: str) -> bool:
    return _failures.get(source_id, 0) >= FAILURE_LIMIT


class Connector(ABC):
    id: str
    category: str = "meta"
    requires_keys: ClassVar[list[str]] = []
    default_cadence: str = "0 * * * *"
    parser_version: int = 1
    rate_limit_per_min: float = 60
    min_gap_s: float = 0.0
    scrapey: bool = False  # honours SCRAPE_ENABLED kill switch

    def is_available(self, settings: Settings) -> bool:
        if self.scrapey and not settings.scrape_enabled:
            return False
        return settings.has_key(*self.requires_keys) if self.requires_keys else True

    def unavailable_reason(self, settings: Settings) -> str | None:
        if self.scrapey and not settings.scrape_enabled:
            return "SCRAPE_ENABLED=false"
        missing = [k for k in self.requires_keys if not getattr(settings, k, "")]
        return f"no key configured: {', '.join(k.upper() for k in missing)}" if missing else None

    @abstractmethod
    async def fetch(self, ctx: IngestContext) -> AsyncIterator[RawDoc]:
        """Yield raw payloads only. Never touches normalised tables."""
        raise NotImplementedError

    def parse(self, doc: RawDoc) -> ParsedBatch:
        """Pure function of a raw doc. Default: archive only."""
        return ParsedBatch()


# --- HTTP ----------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        s = get_settings()
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": s.fpl_user_agent,
                "Accept-Language": "en-GB,en;q=0.9",
            },
            proxy=s.http_proxy_url or None,
        )
    return _client


async def close_http() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def fetch_url(
    url: str,
    source_id: str,
    *,
    per_min: float = 60,
    min_gap_s: float = 0.0,
    retries: int = 3,
    headers: dict | None = None,
    params: dict | None = None,
    method: str = "GET",
    json_body: dict | None = None,
) -> httpx.Response:
    """Rate-limited GET with exponential backoff + jitter, honouring Retry-After."""
    if circuit_open(source_id):
        raise CircuitOpen(f"{source_id} circuit open after {FAILURE_LIMIT} failures")
    await limiter_for(source_id, per_min, min_gap_s).acquire()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = await http_client().request(
                method, url, headers=headers, params=params, json=json_body
            )
            if r.status_code in (429, 503):
                delay = float(r.headers.get("Retry-After", 2 ** attempt))
                await asyncio.sleep(min(delay, 60) + random.uniform(0, 1))
                continue
            r.raise_for_status()
            record_success(source_id)
            return r
        except Exception as e:  # noqa: BLE001 - retried, then surfaced
            last = e
            await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
    record_failure(source_id)
    raise last or RuntimeError(f"{url} failed")


# --- Archive + dedup -----------------------------------------------------------

INLINE_LIMIT = 64 * 1024


def _store_blob(hash_: str, data: bytes) -> str:
    raw = get_settings().raw_dir / hash_[:2]
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{hash_}.json.zst"
    if not path.exists():
        try:
            import zstandard as zstd

            path.write_bytes(zstd.ZstdCompressor(level=10).compress(data))
        except ImportError:
            path = path.with_suffix("")  # .json
            path.write_bytes(data)
    return str(path.relative_to(get_settings().data_dir))


def load_payload(row) -> Any:
    if row["payload_inline"] is not None:
        try:
            return json.loads(row["payload_inline"])
        except json.JSONDecodeError:
            return row["payload_inline"]
    path = get_settings().data_dir / row["payload_path"]
    data = path.read_bytes()
    if path.suffix == ".zst":
        import zstandard as zstd

        data = zstd.ZstdDecompressor().decompress(data)
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data.decode("utf-8", "replace")


def archive(conn, source_id: str, doc: RawDoc, run_id: int | None) -> tuple[int, bool]:
    """Persist one doc under the four-layer dedup strategy. Returns (raw_doc_id, is_new)."""
    h = content_hash(doc.payload)
    existing = query_one(
        "SELECT id FROM raw_documents WHERE source_id=? AND content_hash=?", (source_id, h)
    )
    if existing:
        conn.execute(
            "UPDATE raw_documents SET seen_count=seen_count+1, last_seen_at=? WHERE id=?",
            (utcnow(), existing["id"]),
        )
        return existing["id"], False

    body = (
        json.dumps(doc.payload, separators=(",", ":"), default=str)
        if isinstance(doc.payload, (dict, list))
        else str(doc.payload)
    )
    data = body.encode("utf-8")
    inline, path = (body, None) if len(data) <= INLINE_LIMIT else (None, _store_blob(h, data))

    # Layer 1: same external id, different content -> new row superseding the old.
    supersedes = None
    if doc.external_id:
        prev = query_one(
            "SELECT id FROM raw_documents WHERE source_id=? AND doc_type=? AND external_id=? "
            "ORDER BY id DESC LIMIT 1",
            (source_id, doc.doc_type, str(doc.external_id)),
        )
        supersedes = prev["id"] if prev else None

    # Layer 3: near-duplicate wire copy.
    sh = simhash64(doc.text_for_simhash) if doc.text_for_simhash else None
    meta = dict(doc.meta)
    if sh is not None:
        near = query_one(
            "SELECT id, simhash FROM raw_documents WHERE simhash IS NOT NULL "
            "AND fetched_at > datetime('now','-7 day') "
            "ORDER BY abs(simhash - ?) LIMIT 1",
            (sh,),
        )
        candidates = query(
            "SELECT id, simhash FROM raw_documents WHERE simhash IS NOT NULL "
            "AND fetched_at > datetime('now','-7 day') LIMIT 2000"
        ) if near else []
        for c in candidates:
            if hamming(sh, c["simhash"]) <= NEAR_DUPE_DISTANCE:
                meta["near_dupe_of"] = c["id"]
                break

    cur = conn.execute(
        "INSERT INTO raw_documents(source_id,doc_type,external_id,url,content_hash,simhash,"
        "payload_inline,payload_path,content_bytes,published_at,fetched_at,first_seen_run,"
        "last_seen_at,supersedes_id,meta_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            source_id, doc.doc_type, str(doc.external_id) if doc.external_id else None, doc.url,
            h, sh, inline, path, len(data), doc.published_at, utcnow(), run_id, utcnow(),
            supersedes, jdump(meta),
        ),
    )
    return cur.lastrowid, True


def apply_batch(conn, batch: ParsedBatch) -> int:
    from ..db.engine import upsert_many

    n = 0
    for table, (rows, keys) in batch.tables.items():
        n += upsert_many(conn, table, rows, keys)
    for fn in batch.deferred:
        n += fn(conn) or 0
    return n


# --- Run accounting ------------------------------------------------------------


def start_run(source_id: str, job_name: str, params: dict | None = None) -> int:
    with writer() as conn:
        cur = conn.execute(
            "INSERT INTO ingest_runs(source_id,job_name,started_at,status,params_json) "
            "VALUES(?,?,?,'running',?)",
            (source_id, job_name, utcnow(), jdump(params or {})),
        )
        return cur.lastrowid


def finish_run(run_id: int, result: IngestResult) -> None:
    with writer() as conn:
        conn.execute(
            "UPDATE ingest_runs SET finished_at=?, status=?, requests_made=?, docs_new=?, "
            "docs_duplicate=?, rows_upserted=?, error_text=? WHERE id=?",
            (
                utcnow(), result.status, result.requests_made, result.docs_new,
                result.docs_duplicate, result.rows_upserted, result.error, run_id,
            ),
        )


async def run_connector(
    connector: Connector, season_id: str | None = None, params: dict | None = None
) -> IngestResult:
    """Fetch -> archive+dedup -> parse (new or stale-parser docs only) -> upsert."""
    settings = get_settings()
    result = IngestResult(source_id=connector.id)
    reason = connector.unavailable_reason(settings)
    if reason:
        log.warning("connector %s disabled: %s", connector.id, reason)
        result.status = "skipped"
        result.error = reason
        return result

    run_id = start_run(connector.id, connector.id, params)
    ctx = IngestContext(
        settings=settings,
        run_id=run_id,
        season_id=season_id or settings.current_season,
        params=params or {},
        client=http_client(),
    )
    try:
        async for doc in connector.fetch(ctx):
            result.requests_made += 1
            with writer() as conn:
                doc_id, is_new = archive(conn, connector.id, doc, run_id)
                if is_new:
                    result.docs_new += 1
                else:
                    result.docs_duplicate += 1
                    row = query_one(
                        "SELECT parser_version, parse_status FROM raw_documents WHERE id=?",
                        (doc_id,),
                    )
                    if row and row["parse_status"] == "ok" and \
                            row["parser_version"] == connector.parser_version:
                        continue
                try:
                    batch = connector.parse(doc)
                    batch_rows = apply_batch(conn, batch)
                    result.rows_upserted += batch_rows
                    conn.execute(
                        "UPDATE raw_documents SET parsed_at=?, parser_version=?, "
                        "parse_status='ok', parse_error=NULL WHERE id=?",
                        (utcnow(), connector.parser_version, doc_id),
                    )
                except Exception as e:
                    log.exception("parse failed for %s doc %s", connector.id, doc_id)
                    result.status = "partial"
                    conn.execute(
                        "UPDATE raw_documents SET parse_status='failed', parse_error=? WHERE id=?",
                        (str(e)[:500], doc_id),
                    )
        # Two ways a run can be a silent no-op, and both used to record a clean `ok`.
        #
        # `understat`, `fbref` and `setpieces` each logged success with zero requests, zero
        # documents and zero rows, so every dashboard read healthy while 2026-27 got no xG,
        # no shot volumes and no set-piece duties at all — those features sat flat at zero
        # in the store for a fortnight and nothing anywhere said why. Then `understat`
        # showed the subtler variant: it fetched its page fine and parsed it into nothing.
        #
        # A connector that overrides `parse` exists to write normalised rows; one that does
        # not is an archive-only feed (rss_news, transcripts) whose normal day is zero rows.
        parses_rows = type(connector).parse is not Connector.parse
        empty_reason = None
        if result.requests_made == 0:
            empty_reason = "fetched nothing — the source may have moved or expired"
        elif parses_rows and result.docs_new and not result.rows_upserted:
            empty_reason = (f"fetched {result.docs_new} new document(s) and parsed 0 rows "
                            f"— the feed's shape has probably changed")

        if empty_reason:
            # Deliberately does not reset the failure counter: a source quietly returning
            # nothing must still trip FAILURE_LIMIT rather than look healthy forever.
            result.status = "empty"
            log.warning("connector %s %s", connector.id, empty_reason)
            if record_failure(connector.id) >= FAILURE_LIMIT:
                from ..notify.discord import notify

                await notify(f"⚠️ Source `{connector.id}` has produced nothing "
                             f"{FAILURE_LIMIT} times running.")
        else:
            record_success(connector.id)
    except Exception as e:
        log.exception("connector %s failed", connector.id)
        result.status = "failed"
        result.error = str(e)[:1000]
        if record_failure(connector.id) >= FAILURE_LIMIT:
            from ..notify.discord import notify

            await notify(f"⚠️ Source `{connector.id}` auto-disabled after {FAILURE_LIMIT} failures.")
    finally:
        finish_run(run_id, result)
    return result


async def reprocess(connector: Connector, limit: int | None = None, force: bool = False) -> int:
    """Re-parse the archive without refetching — the whole point of storing raw docs.

    Normally only docs with a stale `parser_version` or a failed parse are touched.
    `force` re-parses everything, which is what you want when reference data the parser
    reads (team names, alias tables, stadium coordinates) has changed rather than the
    parser itself.
    """
    sql = "SELECT * FROM raw_documents WHERE source_id=?"
    if not force:
        sql += " AND (parser_version IS NULL OR parser_version < ? OR parse_status != 'ok')"
    params: tuple = (connector.id,) if force else (connector.id, connector.parser_version)
    if limit:
        sql += " LIMIT ?"
        params = (*params, limit)
    rows = query(sql, params)
    n = 0
    for row in rows:
        doc = RawDoc(
            doc_type=row["doc_type"],
            payload=load_payload(row),
            external_id=row["external_id"],
            url=row["url"],
            published_at=row["published_at"],
            meta=json.loads(row["meta_json"]),
        )
        with writer() as conn:
            try:
                n += apply_batch(conn, connector.parse(doc))
                conn.execute(
                    "UPDATE raw_documents SET parsed_at=?, parser_version=?, parse_status='ok' "
                    "WHERE id=?",
                    (utcnow(), connector.parser_version, row["id"]),
                )
            except Exception as e:  # noqa: BLE001
                conn.execute(
                    "UPDATE raw_documents SET parse_status='failed', parse_error=? WHERE id=?",
                    (str(e)[:500], row["id"]),
                )
    return n


def raw_dir_path(*parts: str) -> Path:
    p = get_settings().raw_dir.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
