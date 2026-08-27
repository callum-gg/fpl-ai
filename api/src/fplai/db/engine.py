"""SQLite engine: pragmas, single-writer lock, schema application, vec extension.

ponytail: plain `schema.sql` applied idempotently instead of Alembic. One file, one
`executescript`, no migration graph to maintain for a single-user local DB. Additive
columns go in `_MIGRATIONS` below; swap in Alembic if the schema ever needs branching.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def utcnow() -> str:
    """The one timestamp format the whole app writes.

    Every leakage cutoff in the feature store is a *string* comparison against a stored
    timestamp, so naive and offset-aware ISO strings must never be mixed: they denote the
    same instant but sort differently.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 30000",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 268435456",
)

# Additive schema changes made after a release. Each is (id, sql); failures on
# "duplicate column" are swallowed so re-running is safe.
_MIGRATIONS: list[tuple[str, str]] = []

_write_lock = threading.Lock()
_local = threading.local()
_vec_available: bool | None = None


def _connect(path: Path, read_only: bool = False) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only and path.exists():
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    for p in PRAGMAS:
        try:
            conn.execute(p)
        except sqlite3.OperationalError:
            pass  # read-only connections reject journal_mode changes
    return conn


def _load_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec if installed. Absence is fine — search falls back to numpy."""
    global _vec_available
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        dim = get_settings().embedding_dim
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0"
            f"(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        _vec_available = True
    except Exception:
        _vec_available = False
    return _vec_available


def vec_available() -> bool:
    if _vec_available is None:
        get_conn()
    return bool(_vec_available)


def get_conn() -> sqlite3.Connection:
    """Thread-local connection. Reads are free; writes must go through writer()."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect(get_settings().db_path)
        _load_vec(conn)
        _local.conn = conn
    return conn


def init_db() -> None:
    conn = _connect(get_settings().db_path)
    _load_vec(conn)
    with _write_lock:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        for _id, sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
    conn.close()
    from .seed import seed_all

    seed_all()


@contextmanager
def writer() -> Iterator[sqlite3.Connection]:
    """All writes funnel through here (01-architecture.md: single writer, zero lock errors)."""
    conn = get_conn()
    with _write_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")


def query(sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
    return get_conn().execute(sql, params).fetchall()


def query_one(sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
    return get_conn().execute(sql, params).fetchone()


def scalar(sql: str, params: tuple | dict = (), default=None):
    row = query_one(sql, params)
    return row[0] if row is not None and row[0] is not None else default


def rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


def upsert(conn: sqlite3.Connection, table: str, row: dict, keys: list[str]) -> None:
    """INSERT ... ON CONFLICT(keys) DO UPDATE for every non-key column."""
    cols = list(row)
    updates = [c for c in cols if c not in keys]
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
        f"ON CONFLICT({','.join(keys)}) DO UPDATE SET "
        + (",".join(f"{c}=excluded.{c}" for c in updates) if updates else f"{keys[0]}={keys[0]}")
    )
    conn.execute(sql, [row[c] for c in cols])


def upsert_many(conn: sqlite3.Connection, table: str, rows: list[dict], keys: list[str]) -> int:
    if not rows:
        return 0
    cols = list(rows[0])
    updates = [c for c in cols if c not in keys]
    sql = (
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
        f"ON CONFLICT({','.join(keys)}) DO UPDATE SET "
        + (",".join(f"{c}=excluded.{c}" for c in updates) if updates else f"{keys[0]}={keys[0]}")
    )
    conn.executemany(sql, [[r.get(c) for c in cols] for r in rows])
    return len(rows)


def duckdb_scan(sql: str):
    """Analytical scans over the same file, per 01-architecture.md. Optional dependency."""
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{get_settings().db_path}' AS s (TYPE sqlite, READ_ONLY)")
    return con.execute(sql).fetchdf()


def jdump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)
