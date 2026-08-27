"""Test fixtures. Nothing here touches the real internet (docs/12: 'what isn't tested')."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api" / "src"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def _isolated_db(tmp_path_factory):
    """Every test run gets its own database, seeded from schema.sql."""
    data_dir = tmp_path_factory.mktemp("fplai-data")
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["DATABASE_URL"] = f"sqlite:///{data_dir / 'test.db'}"
    os.environ["CURRENT_SEASON"] = "2026-27"
    os.environ["APSCHEDULER_ENABLED"] = "false"
    os.environ["LLM_API_KEY"] = ""
    os.environ["DISCORD_WEBHOOK_URL"] = ""

    from fplai.config import get_settings

    get_settings.cache_clear()
    from fplai.db.engine import init_db

    init_db()
    yield data_dir


@pytest.fixture
def db():
    from fplai.db.engine import get_conn

    return get_conn()


def load_fixture(name: str):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"golden fixture {name} not captured; run `fplai capture-fixtures`")
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else text


@pytest.fixture
def bootstrap():
    return load_fixture("bootstrap_static.json")


@pytest.fixture
def seeded_season(db):
    """A minimal but complete season: 2 teams, 4 players, 1 finished fixture."""
    from fplai.db.engine import writer

    with writer() as conn:
        conn.execute("INSERT OR IGNORE INTO seasons(id, is_current) VALUES('2026-27', 1)")
        for fpl_id, name, short in ((1, "Alpha FC", "ALP"), (2, "Beta United", "BET")):
            conn.execute(
                "INSERT OR IGNORE INTO teams(season_id,fpl_team_id,name,short_name) "
                "VALUES('2026-27',?,?,?)",
                (fpl_id, name, short),
            )
        conn.execute(
            "INSERT OR IGNORE INTO gameweeks(season_id,gameweek,deadline_utc,is_next) "
            "VALUES('2026-27',1,'2026-08-22T10:00:00+00:00',1)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO gameweeks(season_id,gameweek,deadline_utc) "
            "VALUES('2026-27',2,'2026-08-29T10:00:00+00:00')"
        )
    return "2026-27"
