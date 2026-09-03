"""The set squad is what you own; drafts and accepted plans must never impersonate it."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from fplai.db.engine import query_one, writer
from fplai.optimiser import recommend as rec_mod

SEASON = "2026-27"


@pytest.fixture
def client(db):
    from fplai.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def squad(db):
    """A squad with a legal 15: 2/5/5/3, three per club, all priced."""
    quota = [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]
    picks, pid = [], 0
    with writer() as conn:
        conn.execute("INSERT OR IGNORE INTO seasons(id) VALUES(?)", (SEASON,))
        for team in range(1, 7):
            conn.execute(
                "INSERT OR REPLACE INTO teams(id,season_id,name,short_name) VALUES(?,?,?,?)",
                (team, SEASON, f"Team {team}", f"TM{team}"),
            )
        conn.execute(
            "INSERT OR REPLACE INTO squads(id,season_id,name) VALUES(1,?,'S')", (SEASON,)
        )
        for position, want in quota:
            for i in range(want):
                pid += 1
                conn.execute(
                    "INSERT OR REPLACE INTO players(id,canonical_name,web_name) VALUES(?,?,?)",
                    (pid, f"P{pid}", f"P{pid}"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO player_seasons(player_id,season_id,position,team_id) "
                    "VALUES(?,?,?,?)",
                    (pid, SEASON, position, pid % 6 + 1),  # <=3 per club across 15 picks
                )
                conn.execute(
                    "INSERT OR REPLACE INTO player_prices(player_id,season_id,price,observed_at) "
                    "VALUES(?,?,?, '2026-08-01T00:00:00+00:00')",
                    (pid, SEASON, 40),
                )
                picks.append({"player_id": pid, "purchase_price": 40})
    return picks


def _set_squad(client, picks):
    r = client.put(
        "/api/squads/1/state",
        json={"gameweek": 1, "bank": 400, "free_transfers": 1, "picks": picks},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_draft_edits_leave_the_set_squad_alone(client, squad):
    _set_squad(client, squad)
    before = client.get("/api/squads/1").json()["state"]

    client.put("/api/squads/1/draft", json={})
    dropped = squad[0]["player_id"]
    draft = client.patch("/api/squads/1/draft", json={"drop": [dropped]}).json()

    assert dropped not in {p["player_id"] for p in draft["picks"]}
    after = client.get("/api/squads/1").json()["state"]
    assert after["id"] == before["id"]
    assert dropped in {p["player_id"] for p in after["picks"]}


def test_an_accepted_recommendation_does_not_become_the_squad_you_own(client, squad):
    """The whole point of a 'planned' state: it is a plan, not your team."""
    _set_squad(client, squad)
    owned = {p["player_id"] for p in squad}

    with writer() as conn:
        conn.execute(
            "INSERT INTO recommendations(id,squad_id,gameweek,generated_at,variant,kind,"
            "horizon_gws,payload_json) VALUES(7,1,1,'2026-08-01T00:00:00+00:00','balanced',"
            "'transfer',5,?)",
            (json.dumps({"lineup": {"xi": [{"player_id": 99}], "bench_order": [],
                                    "captain": 99, "vice": 99},
                         "squad": [{"player_id": 99, "price": 50}]}),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO players(id,canonical_name) VALUES(99,'New')"
        )

    assert client.post("/api/recommendations/7/accept").status_code == 200
    assert query_one("SELECT id FROM squad_states WHERE squad_id=1 AND source='planned'")

    state = client.get("/api/squads/1").json()["state"]
    assert state["source"] == "manual"
    assert {p["player_id"] for p in state["picks"]} == owned


def test_commit_refuses_an_illegal_squad_then_promotes_a_legal_one(client, squad):
    _set_squad(client, squad)
    client.put("/api/squads/1/draft", json={})
    dropped = squad[0]["player_id"]

    client.patch("/api/squads/1/draft", json={"drop": [dropped]})
    refused = client.post("/api/squads/1/draft/commit")
    assert refused.status_code == 400
    assert "needs 15" in refused.json()["detail"]["error"]["message"]

    client.patch("/api/squads/1/draft", json={"add": [dropped]})
    ok = client.post("/api/squads/1/draft/commit")
    assert ok.status_code == 200 and ok.json()["source"] == "manual"
    # Committing consumes the draft, so the scratch copy cannot linger and drift.
    assert client.get("/api/squads/1/draft").status_code == 404


def test_recommendations_optimise_from_the_draft_when_asked(client, squad):
    _set_squad(client, squad)
    client.put("/api/squads/1/draft", json={})
    dropped = squad[0]["player_id"]
    client.patch("/api/squads/1/draft", json={"drop": [dropped]})

    from_set = rec_mod.working_state(1, use_draft=False)
    from_draft = rec_mod.working_state(1, use_draft=True)

    assert len(from_set["picks"]) == 15
    assert len(from_draft["picks"]) == 14
    assert from_draft["source"] == "draft"


def test_two_states_written_in_the_same_second_supersede_rather_than_collide(client, squad):
    """captured_at is second-resolution, so a rapid re-commit must not 500."""
    _set_squad(client, squad)
    second = _set_squad(client, squad)
    assert second["source"] == "manual" and len(second["picks"]) == 15


def test_adding_a_player_with_no_season_record_is_rejected(client, squad):
    _set_squad(client, squad)
    client.put("/api/squads/1/draft", json={})
    r = client.patch("/api/squads/1/draft", json={"add": [123456]})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "unknown_player"


@pytest.mark.parametrize("view", ["set", "draft"])
def test_every_squad_state_names_its_players_and_keeps_its_slot_numbers(client, squad, view):
    """A pick is rendered by name and club, never by id — and `position` stays the 1-15 slot,
    since clobbering it with 'GK' would scramble the bench order."""
    _set_squad(client, squad)
    picks = (
        client.put("/api/squads/1/draft", json={}).json()["picks"]
        if view == "draft"
        else client.get("/api/squads/1").json()["state"]["picks"]
    )

    assert sorted(p["position"] for p in picks) == list(range(1, 16))
    assert {p["position_name"] for p in picks} == {"GK", "DEF", "MID", "FWD"}
    assert all(p["web_name"] for p in picks)
    assert all(p["team_short"].startswith("TM") for p in picks)


def test_a_disconnected_app_can_be_told_the_truth_by_hand(client, squad):
    """Sync is a guess at what FPL holds. When it is wrong, every number must be typeable.

    A stale free-transfer count prices every hit wrong, a chip the app thinks you still
    hold gets planned into a gameweek you cannot use it in, and neither surfaces as an
    error — they just quietly make the plan wrong. So the working copy takes overrides.
    """
    _set_squad(client, squad)
    client.put("/api/squads/1/draft", json={})

    draft = client.patch(
        "/api/squads/1/draft",
        json={
            "gameweek": 7,
            "bank": 125,
            "free_transfers": 4,
            "chip_active": "bboost",
            "chips_used": [{"name": "wildcard", "gameweek": 3}],
        },
    ).json()

    assert (draft["gameweek"], draft["bank"], draft["free_transfers"]) == (7, 125, 4)
    assert draft["chip_active"] == "bboost"
    assert json.loads(draft["chips_used_json"]) == [{"name": "wildcard", "gameweek": 3}]

    # Omitted fields hold their value; "" is the only way to say "no chip this week".
    kept = client.patch("/api/squads/1/draft", json={"bank": 130}).json()
    assert kept["chip_active"] == "bboost" and kept["free_transfers"] == 4
    assert client.patch("/api/squads/1/draft", json={"chip_active": ""}).json()["chip_active"] is None


def test_a_hand_entered_purchase_price_drives_the_selling_price(client, squad):
    """You keep half of any rise, so what you paid is not what he is worth to you.

    Rebuilding a squad at today's prices overstates the budget by exactly the profit the
    game does not let you keep, and the planner then spends money that does not exist.
    """
    _set_squad(client, squad)
    client.put("/api/squads/1/draft", json={})
    pid = squad[0]["player_id"]        # priced at 40 today

    draft = client.patch("/api/squads/1/draft", json={"prices": {str(pid): 34}}).json()
    pick = next(p for p in draft["picks"] if p["player_id"] == pid)
    assert pick["purchase_price"] == 34
    assert pick["selling_price"] == 37   # 3.4 paid, 4.0 now, half the 0.6 rise kept

    assert client.patch("/api/squads/1/draft", json={"prices": {str(pid): 0}}).status_code == 400


def test_nonsense_overrides_are_refused_rather_than_stored(client, squad):
    """These all fail silently downstream, so they have to fail loudly here."""
    _set_squad(client, squad)
    client.put("/api/squads/1/draft", json={})

    assert client.patch("/api/squads/1/draft", json={"free_transfers": 9}).status_code == 400
    assert client.patch("/api/squads/1/draft", json={"chip_active": "triple"}).status_code == 400
    assert client.patch(
        "/api/squads/1/draft", json={"chips_used": [{"name": "wildcard"}]}
    ).status_code == 400
    assert client.patch(
        "/api/squads/1/draft", json={"chips_used": [{"name": "nope", "gameweek": 2}]}
    ).status_code == 400


def test_a_squad_can_be_built_from_nothing_when_there_is_no_sync(client, squad):
    """The disconnected case is the one that needs the editor, not the one to refuse it."""
    r = client.put("/api/squads/1/draft", json={})   # no set squad exists yet
    assert r.status_code == 200, r.text
    assert r.json()["picks"] == [] and r.json()["ok"] is False

    added = client.patch(
        "/api/squads/1/draft", json={"add": [p["player_id"] for p in squad]}
    ).json()
    assert len(added["picks"]) == 15 and added["ok"] is True
    assert client.post("/api/squads/1/draft/commit").status_code == 200
