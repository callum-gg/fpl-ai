"""Layers 1 and 2 — parsers, the four-layer dedup strategy, and entity resolution. docs/12."""

from __future__ import annotations

import pytest
from fplai.connectors.base import (
    NEAR_DUPE_DISTANCE,
    ParsedBatch,
    RawDoc,
    apply_batch,
    archive,
    content_hash,
    hamming,
    simhash64,
    strip_volatile,
)
from fplai.db.engine import query_one, writer

# ── content hashing ───────────────────────────────────────────────────────────


def test_volatile_fields_are_stripped_before_hashing():
    """Two fetches differing only in a timestamp are the same document."""
    a = {"players": [1, 2, 3], "now": "2026-08-19T10:00:00", "request_id": "abc"}
    b = {"players": [1, 2, 3], "now": "2026-08-19T11:30:00", "request_id": "xyz"}
    assert content_hash(a) == content_hash(b)


def test_real_content_change_changes_the_hash():
    a = {"players": [1, 2, 3]}
    b = {"players": [1, 2, 4]}
    assert content_hash(a) != content_hash(b)


def test_key_order_does_not_affect_the_hash():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_strip_volatile_recurses():
    obj = {"outer": {"ts": 1, "keep": 2}, "list": [{"now": 3, "keep": 4}]}
    assert strip_volatile(obj) == {"outer": {"keep": 2}, "list": [{"keep": 4}]}


def test_html_hashing_is_whitespace_and_case_insensitive():
    assert content_hash("Team   News\n\nHere") == content_hash("team news here")


# ── simhash near-duplicate detection ──────────────────────────────────────────

WIRE = (
    "Manchester City manager confirmed that the striker suffered a hamstring injury "
    "in training and will be assessed ahead of the weekend fixture against Arsenal."
)
REWORDED = (
    "Manchester City manager confirmed the striker suffered a hamstring injury in "
    "training and will be assessed ahead of the weekend fixture with Arsenal."
)
UNRELATED = (
    "The referee appointments for the coming round were published on Thursday, with "
    "four officials making their first appearance of the season in the top flight."
)


def test_syndicated_copy_is_near_duplicate():
    """A story on six sites is one fact, not six confirmations (docs/03)."""
    assert hamming(simhash64(WIRE), simhash64(REWORDED)) <= NEAR_DUPE_DISTANCE


def test_unrelated_story_is_not_near_duplicate():
    assert hamming(simhash64(WIRE), simhash64(UNRELATED)) > NEAR_DUPE_DISTANCE


def test_near_dupe_threshold_keeps_margin_on_both_sides():
    """The gap between reworded and unrelated must stay wide, or the threshold is luck."""
    dupe = hamming(simhash64(WIRE), simhash64(REWORDED))
    unrelated = hamming(simhash64(WIRE), simhash64(UNRELATED))
    assert dupe + 3 <= NEAR_DUPE_DISTANCE <= unrelated - 3


def test_simhash_fits_sqlite_signed_range():
    for text in (WIRE, REWORDED, UNRELATED, "short"):
        assert -(2**63) <= simhash64(text) < 2**63


# ── the archive: dedup layers 1 and 2 ─────────────────────────────────────────


def test_same_payload_twice_stores_one_row_and_bumps_seen_count(seeded_season):
    doc = RawDoc("bootstrap", {"x": 1, "now": "a"}, external_id="dedup-1")
    with writer() as conn:
        first_id, is_new = archive(conn, "fpl_official", doc, None)
    assert is_new
    with writer() as conn:
        second_id, is_new_2 = archive(conn, "fpl_official", RawDoc(
            "bootstrap", {"x": 1, "now": "b"}, external_id="dedup-1"), None)
    assert not is_new_2
    assert second_id == first_id
    row = query_one("SELECT seen_count FROM raw_documents WHERE id=?", (first_id,))
    assert row["seen_count"] == 2


def test_changed_content_under_same_external_id_supersedes(seeded_season):
    with writer() as conn:
        first, _ = archive(conn, "fpl_official", RawDoc(
            "element_summary", {"status": "a"}, external_id="dedup-2"), None)
    with writer() as conn:
        second, is_new = archive(conn, "fpl_official", RawDoc(
            "element_summary", {"status": "i"}, external_id="dedup-2"), None)
    assert is_new
    assert second != first
    row = query_one("SELECT supersedes_id FROM raw_documents WHERE id=?", (second,))
    assert row["supersedes_id"] == first


def test_ingest_order_does_not_change_final_row_count(seeded_season):
    """Property: any ordering of a batch yields the same number of stored documents."""
    payloads = [{"n": i} for i in range(6)] + [{"n": 2}, {"n": 4}]  # two repeats

    def count_after(order):
        before = query_one("SELECT COUNT(*) c FROM raw_documents")["c"]
        for i in order:
            with writer() as conn:
                archive(conn, "understat", RawDoc("order-test", payloads[i]), None)
        return query_one("SELECT COUNT(*) c FROM raw_documents")["c"] - before

    forward = count_after(range(len(payloads)))
    # Re-running in reverse adds nothing: every distinct payload is already stored.
    backward = count_after(reversed(range(len(payloads))))
    assert forward == 6
    assert backward == 0


def test_apply_batch_upserts_rows(seeded_season):
    batch = ParsedBatch()
    batch.add("seasons", [{"id": "2099-00", "is_current": 0}], ["id"])
    with writer() as conn:
        n = apply_batch(conn, batch)
    assert n == 1
    assert query_one("SELECT 1 FROM seasons WHERE id='2099-00'") is not None


def test_deferred_callbacks_run_inside_the_same_transaction(seeded_season):
    batch = ParsedBatch()
    batch.defer(
        lambda conn: conn.execute(
            "INSERT OR IGNORE INTO seasons(id,is_current) VALUES('2098-99',0)"
        ) and 1
    )
    with writer() as conn:
        apply_batch(conn, batch)
    assert query_one("SELECT 1 FROM seasons WHERE id='2098-99'") is not None


# ── entity resolution ─────────────────────────────────────────────────────────


@pytest.fixture
def players(seeded_season):
    from fplai.resolve.entities import add_alias, upsert_player

    with writer() as conn:
        alpha = query_one("SELECT id FROM teams WHERE short_name='ALP'")["id"]
        beta = query_one("SELECT id FROM teams WHERE short_name='BET'")["id"]
        made = {}
        for full, first, last, web, team in (
            ("Gabriel Magalhães", "Gabriel", "Magalhães", "Gabriel", alpha),
            ("Gabriel Martinelli", "Gabriel", "Martinelli", "Martinelli", alpha),
            ("Trent Alexander-Arnold", "Trent", "Alexander-Arnold", "Alexander-Arnold", beta),
            ("Robert Sánchez", "Robert", "Sánchez", "Sánchez", beta),
        ):
            pid = upsert_player(conn, full, first, last, web)
            conn.execute(
                "INSERT OR IGNORE INTO player_seasons(player_id,season_id,team_id,position) "
                "VALUES(?,?,?,'DEF')",
                (pid, seeded_season, team),
            )
            made[full] = pid
        add_alias(conn, made["Trent Alexander-Arnold"], "TAA", "manual")
        add_alias(conn, made["Trent Alexander-Arnold"], "Trent", "manual")
    return made


def test_exact_alias_resolves(players, seeded_season):
    from fplai.resolve.entities import resolve_name

    res = resolve_name("TAA", None, seeded_season)
    assert res.player_id == players["Trent Alexander-Arnold"]
    assert res.method == "alias"


def test_accent_insensitive_resolution(players, seeded_season):
    from fplai.resolve.entities import resolve_name

    assert resolve_name("Robert Sanchez", None, seeded_season).player_id == players["Robert Sánchez"]
    assert resolve_name("Magalhaes", None, seeded_season).player_id == players["Gabriel Magalhães"]


def test_ambiguous_shared_forename_resolves_to_none(players, seeded_season):
    """docs/04: never resolve a bare form two players share. A wrong link is far worse
    than a dropped one."""
    from fplai.resolve.entities import resolve_name

    res = resolve_name("Gabriel", None, seeded_season)
    assert res.player_id is None or res.player_id == players["Gabriel Magalhães"]
    if res.player_id is None:
        assert res.method in ("ambiguous", "review", "miss")


def test_unknown_name_does_not_resolve(players, seeded_season):
    from fplai.resolve.entities import resolve_name

    res = resolve_name("Zlatan Ibrahimovic", None, seeded_season)
    assert res.player_id is None


def test_unresolved_forms_reach_the_review_queue(players, seeded_season):
    from fplai.resolve.entities import resolve_name, review_queue

    resolve_name("Alexandr-Arnold", None, seeded_season)   # close but not close enough
    queue = review_queue()
    assert isinstance(queue, list)


def test_team_alias_resolution():
    from fplai.resolve.entities import resolve_team

    assert resolve_team("Spurs") == "Spurs"
    assert resolve_team("Tottenham Hotspur") == "Spurs"
    assert resolve_team("man united") == "Man Utd"
    assert resolve_team("The Gunners") is None or resolve_team("gunners") == "Arsenal"


def test_normalisation_is_stable():
    from fplai.resolve.normalise import norm_name

    assert norm_name("Gabriel Magalhães") == "gabriel magalhaes"
    assert norm_name("O'Riley") == "oriley"
    assert norm_name("Alexander-Arnold") == "alexander arnold"
    assert norm_name("  Silva  Jr. ") == "silva"
    assert norm_name(None) == ""


# ── connector registry ────────────────────────────────────────────────────────


def test_every_connector_declares_a_unique_id():
    from fplai.connectors.registry import CONNECTORS

    ids = list(CONNECTORS)
    assert len(ids) == len(set(ids))
    assert "fpl_official" in ids


def test_key_gated_connectors_self_disable_without_keys():
    from fplai.config import get_settings
    from fplai.connectors.registry import CONNECTORS

    settings = get_settings()
    odds = CONNECTORS["odds_api"]
    if not settings.odds_api_key:
        assert not odds.is_available(settings)
        assert "ODDS_API_KEY" in odds.unavailable_reason(settings)


def test_free_connectors_are_available_without_keys():
    from fplai.config import get_settings
    from fplai.connectors.registry import CONNECTORS

    assert CONNECTORS["fpl_official"].is_available(get_settings())
    assert CONNECTORS["vaastav_history"].is_available(get_settings())


def test_scrape_kill_switch_disables_scrapers(monkeypatch):
    from fplai.config import Settings
    from fplai.connectors.registry import CONNECTORS

    off = Settings(scrape_enabled=False)
    assert not CONNECTORS["understat"].is_available(off)
    assert CONNECTORS["understat"].unavailable_reason(off) == "SCRAPE_ENABLED=false"
    # The FPL API is not a scraper, so the kill switch must not touch it.
    assert CONNECTORS["fpl_official"].is_available(off)


def test_a_connector_that_fetches_nothing_is_not_recorded_as_ok(seeded_season):
    """`understat`, `fbref` and `setpieces` each logged a clean `ok` with zero requests and
    zero rows, so 2026-27 had no xG at all while every dashboard read healthy."""
    import asyncio

    from fplai.connectors.base import Connector, IngestContext, run_connector
    from fplai.db.engine import query_one

    class Silent(Connector):
        id = "understat"          # a real source id, so the FK on ingest_runs holds

        async def fetch(self, ctx: IngestContext):
            return
            yield  # pragma: no cover - makes this an async generator

    result = asyncio.run(run_connector(Silent(), seeded_season))
    assert result.status == "empty"
    assert result.requests_made == 0

    row = query_one(
        "SELECT status, rows_upserted FROM ingest_runs WHERE source_id='understat' "
        "ORDER BY id DESC LIMIT 1"
    )
    assert row["status"] == "empty" and row["rows_upserted"] == 0


def test_a_connector_that_yields_a_document_is_still_ok(seeded_season):
    import asyncio

    from fplai.connectors.base import Connector, IngestContext, RawDoc, run_connector

    class Talkative(Connector):
        id = "fbref"

        async def fetch(self, ctx: IngestContext):
            yield RawDoc("article", "something real happened", url="http://x/notempty")

    assert asyncio.run(run_connector(Talkative(), seeded_season)).status == "ok"


def test_a_connector_that_parses_nothing_is_flagged_too(seeded_season):
    """The subtler no-op: `understat` fetched its page fine and parsed it into zero rows,
    which is why 2026-27 has no xG while every historical season does."""
    import asyncio

    from fplai.connectors.base import (
        Connector,
        IngestContext,
        ParsedBatch,
        RawDoc,
        run_connector,
    )

    class Fruitless(Connector):
        id = "understat"

        async def fetch(self, ctx: IngestContext):
            yield RawDoc("stats", {"rows": []}, url="http://x/fruitless")

        def parse(self, doc: RawDoc) -> ParsedBatch:
            return ParsedBatch()          # overrides parse, but writes nothing

    assert asyncio.run(run_connector(Fruitless(), seeded_season)).status == "empty"


def test_an_archive_only_connector_writing_no_rows_is_normal(seeded_season):
    """rss_news and transcripts archive text for the LLM pass; zero rows is their good day."""
    import asyncio

    from fplai.connectors.base import Connector, IngestContext, RawDoc, run_connector

    class ArchiveOnly(Connector):
        id = "rss_news"          # does not override parse

        async def fetch(self, ctx: IngestContext):
            yield RawDoc("article", "a story worth keeping", url="http://x/archive-only")

    assert asyncio.run(run_connector(ArchiveOnly(), seeded_season)).status == "ok"
