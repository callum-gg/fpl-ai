"""European fixtures: ESPN fetch, idempotent parse, and what the congestion features see.

GW12 2026-27: Sunderland and Bournemouth played Europa League games on Thursday 26 Nov
and again on Sunday 29 Nov, but the app only loaded the Champions League. It rated both
clubs six days rested, and each weekly Champions League doc re-inserted every match
(120 rows for 40).
"""

from __future__ import annotations

import asyncio

import pytest

from fplai.connectors.base import IngestContext, RawDoc, apply_batch
from fplai.connectors.euro_fixtures import EuroFixturesConnector
from fplai.db.engine import query, writer

SEASON = "2026-27"
CLUBS = {"Sunderland": "SUN", "Bournemouth": "BOU", "Brighton": "BHA",
         "Arsenal": "ARS", "Man City": "MCI"}


@pytest.fixture
def clubs(seeded_season):
    with writer() as conn:
        for i, (name, short) in enumerate(CLUBS.items(), start=31):
            conn.execute("INSERT OR IGNORE INTO teams(season_id,fpl_team_id,name,short_name) "
                         "VALUES(?,?,?,?)", (SEASON, i, name, short))
        conn.execute("DELETE FROM weather_observations WHERE fixture_id IN "
                     "(SELECT id FROM fixtures WHERE competition != 'PL')")
        conn.execute("DELETE FROM fixtures WHERE competition != 'PL'")
    return {r["short_name"]: r["id"] for r in query(
        "SELECT id, short_name FROM teams WHERE season_id=?", (SEASON,))}


def _event(eid, date, home, away, completed):
    return {"id": eid, "date": date, "home": home, "away": away, "completed": completed}


def _parse(comp, events, season=SEASON):
    doc = RawDoc("euro_matches", {"comp": comp, "events": events},
                 external_id=f"{comp}:{season}")
    with writer() as conn:
        return apply_batch(conn, EuroFixturesConnector().parse(doc))


def _rows(comp):
    return [dict(r) for r in query(
        "SELECT kickoff_utc, home_team_id, away_team_id, finished FROM fixtures "
        "WHERE season_id=? AND competition=? ORDER BY kickoff_utc", (SEASON, comp))]


def test_europa_and_conference_matches_land_as_fixtures(clubs):
    _parse("UEL", [_event("1", "2026-11-26T20:00Z", "Sunderland", "Jagiellonia Bialystok", True),
                   _event("2", "2026-11-26T17:45Z", "Celta Vigo", "AFC Bournemouth", True),
                   _event("3", "2026-11-26T17:45Z", "Besiktas", "Hapoel Be'er", True)])
    _parse("UECL", [_event("4", "2026-11-26T17:45Z", "Brighton & Hove Albion",
                           "Universitatea Craiova", True)])

    uel = _rows("UEL")
    # The foreign-only match is dropped; a PL club against a foreign one is stored against
    # itself (foreign clubs have no `teams` row), whichever side it was on.
    assert [(r["home_team_id"], r["away_team_id"]) for r in uel] == [
        (clubs["BOU"], clubs["BOU"]), (clubs["SUN"], clubs["SUN"])]
    # Normalised to the PL fixtures' form, so string comparison against them holds.
    assert uel[1]["kickoff_utc"] == "2026-11-26T20:00:00+00:00"
    assert all(r["finished"] == 1 for r in uel)
    assert _rows("UECL")[0]["home_team_id"] == clubs["BHA"]


def test_reparsing_replaces_rather_than_duplicates(clubs):
    """Before the match: scheduled. After: finished, and kickoff moved. Still one row."""
    _parse("UCL", [_event("9", "2026-12-09T20:00Z", "Arsenal", "Club Brugge", False)])
    _parse("UCL", [_event("9", "2026-12-09T17:45Z", "Arsenal", "Club Brugge", True)])
    _parse("UCL", [_event("9", "2026-12-09T17:45Z", "Arsenal", "Club Brugge", True)])

    assert _rows("UCL") == [{"kickoff_utc": "2026-12-09T17:45:00+00:00",
                             "home_team_id": clubs["ARS"], "away_team_id": clubs["ARS"],
                             "finished": 1}]


def test_a_forecast_on_a_european_row_does_not_block_the_replace(clubs):
    """Live GW12: the weather job had forecast 35 UCL rows, and the first ESPN parse of
    2026-27 UCL died on the foreign key, leaving all 120 duplicates in place."""
    _parse("UCL", [_event("9", "2026-12-09T20:00Z", "Arsenal", "Club Brugge", False)])
    with writer() as conn:
        conn.execute("INSERT INTO weather_observations(fixture_id,observed_at) "
                     "SELECT id, '2026-11-26T07:00:00+00:00' FROM fixtures WHERE competition='UCL'")
    _parse("UCL", [_event("9", "2026-12-09T20:00Z", "Arsenal", "Club Brugge", True)])

    assert [r["finished"] for r in _rows("UCL")] == [1]
    assert query("SELECT COUNT(*) n FROM weather_observations w JOIN fixtures f "
                 "ON f.id=w.fixture_id WHERE f.competition='UCL'")[0]["n"] == 0


def test_a_parse_only_replaces_its_own_competition_and_season(clubs):
    _parse("UEL", [_event("1", "2026-11-26T20:00Z", "Sunderland", "Jagiellonia Bialystok", True)])
    _parse("UCL", [_event("9", "2026-12-09T20:00Z", "Arsenal", "Club Brugge", False)])
    _parse("UCL", [], season="2025-26")
    _parse("UCL", [_event("10", "2026-12-10T20:00Z", "Manchester City", "Arsenal", False)])

    assert len(_rows("UEL")) == 1
    # Both sides are PL clubs, so it's a real fixture rather than a self-match.
    assert [(r["home_team_id"], r["away_team_id"]) for r in _rows("UCL")] == [
        (clubs["MCI"], clubs["ARS"])]


def test_a_football_data_doc_from_before_espn_parses_to_nothing(clubs):
    _parse("UEL", [_event("1", "2026-11-26T20:00Z", "Sunderland", "Jagiellonia Bialystok", True)])
    legacy = RawDoc("euro_matches", {"code": "CL", "data": {"matches": []}},
                    external_id=f"CL:{SEASON}")
    with writer() as conn:
        assert apply_batch(conn, EuroFixturesConnector().parse(legacy)) == 0
    assert len(_rows("UEL")) == 1


def test_congestion_reads_thursdays_europa_game(clubs):
    """Friday's deadline, Thursday's game: one day's rest, not six, and the following
    Thursday's game counts once toward the next fortnight's load."""
    from fplai.features.build import _days_rest, _team_history, _upcoming

    _parse("UEL", [_event("1", "2026-11-26T20:00Z", "Sunderland", "Jagiellonia Bialystok", True),
                   _event("5", "2026-12-10T20:00Z", "AC Milan", "Sunderland", False)])
    _parse("UEL", [_event("1", "2026-11-26T20:00Z", "Sunderland", "Jagiellonia Bialystok", True),
                   _event("5", "2026-12-10T20:00Z", "AC Milan", "Sunderland", False)])

    deadline = "2026-11-27T18:30:00+00:00"
    history = _team_history(clubs["SUN"], deadline)
    assert history[0]["competition"] == "UEL"
    assert _days_rest(history, deadline) == 0
    assert [f["competition"] for f in _upcoming(clubs["SUN"], deadline)].count("UEL") == 1


# ── fetch ────────────────────────────────────────────────────────────────────


class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _espn(eid, date, home, away, completed=True):
    return {"id": eid, "date": date, "competitions": [{
        "competitors": [{"homeAway": "home", "team": {"displayName": home}},
                        {"homeAway": "away", "team": {"displayName": away}}],
        "status": {"type": {"completed": completed}}}]}


def _teams(*ids):
    return {"items": [{"$ref": f"http://espn/teams/{i}?lang=en"} for i in ids]}


def _fetch(monkeypatch, routes, seasons=None):
    """Run fetch against canned responses; a route mapped to an exception raises it."""
    calls = []

    async def fake(url, source_id, *, params=None, **_):
        key = url.split("/soccer/")[-1]
        if (params or {}).get("fixture"):
            key += "?fixture"
        calls.append(key)
        got = routes.get(key, {"events": []})
        if isinstance(got, Exception):
            raise got
        return _Resp(got)

    monkeypatch.setattr("fplai.connectors.euro_fixtures.fetch_url", fake)
    ctx = IngestContext(settings=None, run_id=0, season_id=SEASON,
                        params={"seasons": seasons} if seasons else {})

    async def collect():
        return [d async for d in EuroFixturesConnector().fetch(ctx)]

    return asyncio.run(collect()), calls


def test_fetch_takes_only_pl_clubs_and_dedupes_shared_matches(monkeypatch):
    city_v_arsenal = _espn("10", "2026-12-10T20:00Z", "Manchester City", "Arsenal", False)
    routes = {
        "leagues/eng.1/seasons/2026/teams": _teams(359, 382, 366),
        "leagues/uefa.champions/seasons/2026/teams": _teams(359, 382, 103),
        "uefa.champions/teams/359/schedule": {"events": [
            _espn("9", "2026-11-24T20:00Z", "Arsenal", "Club Brugge")]},
        "uefa.champions/teams/359/schedule?fixture": {"events": [city_v_arsenal]},
        "uefa.champions/teams/382/schedule?fixture": {"events": [city_v_arsenal]},
        "leagues/uefa.europa/seasons/2026/teams": _teams(366, 85),
        "uefa.europa/teams/366/schedule": {"events": [
            _espn("1", "2026-11-26T20:00Z", "Sunderland", "Jagiellonia Bialystok")]},
        "leagues/uefa.europa.conf/seasons/2026/teams": _teams(500, 501),
    }
    docs, calls = _fetch(monkeypatch, routes)

    by_comp = {d.payload["comp"]: d for d in docs}
    assert [e["id"] for e in by_comp["UCL"].payload["events"]] == ["9", "10"]
    assert by_comp["UCL"].external_id == "UCL:2026-27"
    assert by_comp["UEL"].payload["events"] == [{
        "id": "1", "date": "2026-11-26T20:00Z", "home": "Sunderland",
        "away": "Jagiellonia Bialystok", "completed": True}]
    # No English club in the Conference League: an empty doc, which clears stale rows.
    assert by_comp["UECL"].payload["events"] == []
    # Foreign clubs' schedules (103 Milan, 85 Celta) are never requested.
    assert not any("/teams/103/" in c or "/teams/85/" in c for c in calls)


def test_fetch_keeps_upcoming_games_out_of_past_seasons(monkeypatch):
    """`fixture=true` ignores `season`: a 2023-24 backfill got this season's games."""
    routes = {
        "leagues/eng.1/seasons/2023/teams": _teams(331),
        "leagues/uefa.europa/seasons/2023/teams": _teams(331),
        "uefa.europa/teams/331/schedule": {"events": [
            _espn("7", "2023-12-14T20:00Z", "Marseille", "Brighton & Hove Albion")]},
        "uefa.europa/teams/331/schedule?fixture": {"events": [
            _espn("8", "2026-12-10T17:45Z", "Brighton & Hove Albion", "Ajax", False)]},
    }
    docs, _ = _fetch(monkeypatch, routes, seasons=["2023-24"])
    uel = next(d for d in docs if d.payload["comp"] == "UEL")
    assert [e["id"] for e in uel.payload["events"]] == ["7"]


def test_a_failed_request_skips_that_competition_instead_of_wiping_it(monkeypatch):
    routes = {
        "leagues/eng.1/seasons/2026/teams": _teams(366),
        "leagues/uefa.europa/seasons/2026/teams": _teams(366),
        "uefa.europa/teams/366/schedule?fixture": RuntimeError("503"),
    }
    docs, _ = _fetch(monkeypatch, routes)
    assert {d.payload["comp"] for d in docs} == {"UCL", "UECL"}


def test_no_pl_clubs_from_espn_yields_nothing(monkeypatch):
    docs, _ = _fetch(monkeypatch, {"leagues/eng.1/seasons/2026/teams": {"items": []}})
    assert docs == []
