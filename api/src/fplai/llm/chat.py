"""Tool-calling chat over the app's own data. docs/08 chat.

Every tool here is read-only, and a test asserts the registry exposes no mutating
function. The chat may propose changes; applying one is always a UI action by you.

`run_optimiser` is what makes iterative team-building work: "what if I go without a
premium striker?" becomes a constrained re-solve answered with real numbers.
"""

from __future__ import annotations

import json
import logging

from ..config import get_settings
from ..db.engine import query, query_one
from .client import LLMUnavailable, complete
from .tasks import get as get_task

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6


# --- read-only tool implementations ---------------------------------------------


def search_corpus(query_text: str, k: int = 10, source_id: str | None = None) -> list[dict]:
    from .embed import hybrid_search

    hits = hybrid_search(query_text, k=k, filters={"source_id": source_id} if source_id else None)
    return [
        {"chunk_id": h["id"], "text": h["text"][:600], "source": h["source_id"],
         "title": h.get("title"), "link": h.get("deep_link"), "score": round(h["score"], 4)}
        for h in hits
    ]


def get_player(player_id: int) -> dict | None:
    season = get_settings().current_season
    row = query_one(
        "SELECT p.id, p.canonical_name, p.web_name, ps.position, t.name team, "
        "(SELECT price FROM player_prices WHERE player_id=p.id AND season_id=ps.season_id "
        " ORDER BY observed_at DESC LIMIT 1) price "
        "FROM players p JOIN player_seasons ps ON ps.player_id=p.id AND ps.season_id=? "
        "LEFT JOIN teams t ON t.id=ps.team_id WHERE p.id=?",
        (season, player_id),
    )
    return dict(row) if row else None


def get_prediction(player_id: int, gameweek: int) -> dict | None:
    from ..models.predict import prediction_for

    return prediction_for(player_id, get_settings().current_season, gameweek)


def compare_players(player_ids: list[int], gameweek: int) -> list[dict]:
    out = []
    for pid in player_ids[:8]:
        p = get_player(pid) or {}
        pred = get_prediction(pid, gameweek) or {}
        out.append(
            {
                "player_id": pid,
                "name": p.get("web_name"),
                "team": p.get("team"),
                "price": p.get("price"),
                "exp_points": pred.get("exp_points"),
                "p_start": pred.get("p_start"),
                "p_haul_10": pred.get("p_haul_10"),
                "sd_points": pred.get("sd_points"),
                "prediction_id": pred.get("id"),
            }
        )
    return out


def explain_feature(player_id: int, gameweek: int, name: str | None = None) -> list[dict]:
    from ..features.build import feature_explanations

    rows = feature_explanations(player_id, get_settings().current_season, gameweek)
    return [r for r in rows if name is None or name in r["name"]][:40]


def get_squad_state(squad_id: int) -> dict | None:
    from ..optimiser.recommend import current_state

    state = current_state(squad_id)
    if state is None:
        return None
    return {
        "gameweek": state["gameweek"],
        "bank": state["bank"],
        "free_transfers": state["free_transfers"],
        "squad_value": state["squad_value"],
        "picks": [
            {"player_id": p["player_id"], "position": p["position"],
             "is_captain": bool(p["is_captain"]), "selling_price": p["selling_price"]}
            for p in state["picks"]
        ],
    }


def list_fixtures(team_id: int, n: int = 6) -> list[dict]:
    season = get_settings().current_season
    return [
        dict(r)
        for r in query(
            "SELECT f.gameweek, f.kickoff_utc, f.competition, "
            "CASE WHEN f.home_team_id=? THEN 1 ELSE 0 END is_home, "
            "CASE WHEN f.home_team_id=? THEN ta.name ELSE th.name END opponent, "
            "CASE WHEN f.home_team_id=? THEN f.fdr_home ELSE f.fdr_away END difficulty "
            "FROM fixtures f JOIN teams th ON th.id=f.home_team_id "
            "JOIN teams ta ON ta.id=f.away_team_id "
            "WHERE f.season_id=? AND (f.home_team_id=? OR f.away_team_id=?) "
            "AND f.kickoff_utc >= datetime('now') ORDER BY f.kickoff_utc LIMIT ?",
            (team_id, team_id, team_id, season, team_id, team_id, n),
        )
    ]


def run_optimiser(squad_id: int, force_in: list[int] | None = None,
                  force_out: list[int] | None = None, chip: str | None = None,
                  max_hits: int | None = None) -> dict:
    """Constrained re-solve. The honest answer to 'what if I did X?'."""
    from ..connectors.fpl_official import next_gameweek
    from ..optimiser.recommend import recommend

    season = get_settings().current_season
    recs = recommend(
        squad_id, season, next_gameweek(season), variants=["balanced"],
        constraints={"force_in": force_in or [], "force_out": force_out or [],
                     "chip": chip, "max_hits": max_hits},
        persist=False,
    )
    if not recs:
        return {"error": "no feasible plan under those constraints"}
    payload = recs[0]["payload"]
    return {
        "headline": payload["headline"],
        "transfers": payload["transfers"],
        "totals": payload["totals"],
        "hits": payload["hits"],
        "chip": payload["chip"],
        "delta_vs_do_nothing": payload.get("delta_vs_do_nothing"),
    }


def find_player(name: str) -> list[dict]:
    from ..resolve.normalise import norm_name

    rows = query(
        "SELECT DISTINCT p.id, p.canonical_name, p.web_name FROM players p "
        "JOIN player_aliases a ON a.player_id=p.id WHERE a.alias_norm LIKE ? LIMIT 10",
        (f"%{norm_name(name)}%",),
    )
    return [dict(r) for r in rows]


TOOLS = {
    "search_corpus": search_corpus,
    "get_player": get_player,
    "find_player": find_player,
    "get_prediction": get_prediction,
    "compare_players": compare_players,
    "explain_feature": explain_feature,
    "get_squad_state": get_squad_state,
    "list_fixtures": list_fixtures,
    "run_optimiser": run_optimiser,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Hybrid vector+keyword search over ingested articles, transcripts and posts.",
        "parameters": {"type": "object", "properties": {
            "query_text": {"type": "string"}, "k": {"type": "integer"},
            "source_id": {"type": "string"}}, "required": ["query_text"]}}},
    {"type": "function", "function": {
        "name": "find_player",
        "description": "Resolve a player name or nickname to candidate player ids.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "get_player", "description": "Player metadata: position, club, current price.",
        "parameters": {"type": "object", "properties": {"player_id": {"type": "integer"}},
                       "required": ["player_id"]}}},
    {"type": "function", "function": {
        "name": "get_prediction",
        "description": "The model's projection for one player in one gameweek.",
        "parameters": {"type": "object", "properties": {
            "player_id": {"type": "integer"}, "gameweek": {"type": "integer"}},
            "required": ["player_id", "gameweek"]}}},
    {"type": "function", "function": {
        "name": "compare_players", "description": "Side-by-side projections for several players.",
        "parameters": {"type": "object", "properties": {
            "player_ids": {"type": "array", "items": {"type": "integer"}},
            "gameweek": {"type": "integer"}}, "required": ["player_ids", "gameweek"]}}},
    {"type": "function", "function": {
        "name": "explain_feature",
        "description": "Feature values and percentiles behind a player's projection.",
        "parameters": {"type": "object", "properties": {
            "player_id": {"type": "integer"}, "gameweek": {"type": "integer"},
            "name": {"type": "string"}}, "required": ["player_id", "gameweek"]}}},
    {"type": "function", "function": {
        "name": "get_squad_state", "description": "The squad's current picks, bank and free transfers.",
        "parameters": {"type": "object", "properties": {"squad_id": {"type": "integer"}},
                       "required": ["squad_id"]}}},
    {"type": "function", "function": {
        "name": "list_fixtures", "description": "A team's upcoming fixtures with difficulty.",
        "parameters": {"type": "object", "properties": {
            "team_id": {"type": "integer"}, "n": {"type": "integer"}}, "required": ["team_id"]}}},
    {"type": "function", "function": {
        "name": "run_optimiser",
        "description": "Re-solve the squad under forced constraints and return real numbers.",
        "parameters": {"type": "object", "properties": {
            "squad_id": {"type": "integer"},
            "force_in": {"type": "array", "items": {"type": "integer"}},
            "force_out": {"type": "array", "items": {"type": "integer"}},
            "chip": {"type": "string"}, "max_hits": {"type": "integer"}},
            "required": ["squad_id"]}}},
]


async def chat(messages: list[dict], squad_id: int | None = None):
    """Run the tool loop. Yields step dicts so the UI can show collapsed tool calls."""
    task = get_task("chat")
    convo = list(messages)
    if squad_id is not None:
        convo.insert(0, {"role": "system",
                         "content": f"The active squad id is {squad_id}. Scope answers to it."})

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            resp = await complete(task, convo, tools=TOOL_SCHEMAS)
        except LLMUnavailable as e:
            yield {"type": "error", "message": str(e)}
            return
        except Exception as e:
            log.exception("chat call failed")
            yield {"type": "error", "message": str(e)}
            return

        tool_calls = (resp.data or {}).get("tool_calls") if isinstance(resp.data, dict) else None
        if not tool_calls:
            yield {"type": "message", "content": resp.text}
            return

        convo.append({"role": "assistant", "content": resp.text or None, "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            impl = TOOLS.get(name)
            if impl is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = impl(**args)
                except Exception as e:  # noqa: BLE001 - a bad tool call is an answer too
                    result = {"error": str(e)}
            yield {"type": "tool", "name": name, "args": args,
                   "summary": _summarise(name, result)}
            convo.append({"role": "tool", "tool_call_id": call.get("id"),
                          "content": json.dumps(result, default=str)[:8000]})

    yield {"type": "message", "content": "I ran out of tool steps before reaching an answer."}


def _summarise(name: str, result) -> str:
    if isinstance(result, list):
        return f"{name}: {len(result)} results"
    if isinstance(result, dict) and "error" in result:
        return f"{name}: {result['error']}"
    return f"{name}: ok"


def assert_read_only() -> list[str]:
    """docs/12 layer 7: the tool registry must expose no mutating function."""
    import inspect

    offenders = []
    for name, fn in TOOLS.items():
        src = inspect.getsource(fn)
        if any(kw in src for kw in ("writer(", "INSERT ", "UPDATE ", "DELETE ")):
            offenders.append(name)
    return offenders
