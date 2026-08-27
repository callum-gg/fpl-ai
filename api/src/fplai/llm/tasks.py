"""The task registry: one prompt + schema per task. docs/08.

Claim extraction is the workhorse and the important one. Its rules are baked into the
system prompt: extract only what is *asserted in this chunk*, keep `text_span` a short
verbatim fragment so the evidence panel can anchor it, distinguish the speaker's own
view from them reporting someone else's, and return `[]` freely — most chunks contain no
claims, and a model that always finds something is worse than useless.
"""

from __future__ import annotations

from .client import LLMTask

CLAIM_TYPES = [
    "injury", "return", "rotation", "form", "recommendation", "captain_pick", "avoid",
    "transfer_rumour", "role_change", "penalty_duty", "manager_quote",
]

EXTRACT_CLAIMS_SYSTEM = f"""You extract structured FPL claims from football text.

Return ONLY a JSON object: {{"claims": [...]}}.

Each claim has:
  player        the player's surface form exactly as written, or null if ambiguous
  claim_type    one of {CLAIM_TYPES}
  stance        positive | negative | neutral
  sentiment     -1.0 to 1.0
  confidence    0.0 to 1.0, YOUR confidence that this claim is really asserted
  horizon_gw    the gameweek it refers to, or null
  is_reported   true if the speaker is relaying someone else's claim
                ("Ornstein says..." is a report, not the speaker's own view)
  text_span     a VERBATIM fragment from the chunk, UNDER 15 WORDS

Hard rules:
- Extract only what is asserted IN THIS CHUNK. Never infer from world knowledge.
- text_span must appear verbatim in the input. Never reproduce whole paragraphs.
- If the player is ambiguous or a bare shared surname, set player to the surface form
  and let the resolver decide — do not guess which player is meant.
- Return {{"claims": []}} freely. Most chunks contain no claims at all.

Examples:

Input: "Right, so Haaland's a doubt for the weekend, Pep said in the presser he's got a
knock but they're hopeful."
Output: {{"claims": [
  {{"player": "Haaland", "claim_type": "injury", "stance": "negative", "sentiment": -0.5,
    "confidence": 0.85, "horizon_gw": null, "is_reported": true,
    "text_span": "Haaland's a doubt for the weekend"}}
]}}

Input: "Mate, I cannot believe I captained him. Absolute disaster. Anyway, like and
subscribe, see you next week."
Output: {{"claims": []}}

Input: "I'm bringing in Saka this week, he's on penalties now and the fixtures turn."
Output: {{"claims": [
  {{"player": "Saka", "claim_type": "recommendation", "stance": "positive", "sentiment": 0.8,
    "confidence": 0.9, "horizon_gw": null, "is_reported": false,
    "text_span": "I'm bringing in Saka this week"}},
  {{"player": "Saka", "claim_type": "penalty_duty", "stance": "positive", "sentiment": 0.6,
    "confidence": 0.7, "horizon_gw": null, "is_reported": false,
    "text_span": "he's on penalties now"}}
]}}
"""

RESOLVE_ENTITY_SYSTEM = """You resolve a football player nickname to an id, using context.

You get a surface form, the surrounding sentences, and a candidate list of players
(id, name, club) restricted to those plausibly present in this document.

Return ONLY: {"player_id": <id or null>, "confidence": 0.0-1.0, "reasoning": "one line"}

Return null when the context does not distinguish between candidates. A wrong link is
far worse than a dropped one — it poisons every text feature downstream.
"""

INJURY_SEVERITY_SYSTEM = """You classify football injury reports.

Return ONLY:
{"status": "available|doubt|injured|suspended|unknown",
 "chance_pct": 0-100 or null,
 "expected_gws_out": integer or null,
 "issue": "short body part or reason",
 "confidence": 0.0-1.0}

"Knock, assessed daily" is a doubt, not an injury. "Facing weeks out" is injured with an
expected_gws_out estimate. Say unknown when the text does not support a call.
"""

SUMMARISE_VIDEO_SYSTEM = """Summarise an FPL video for someone who has not watched it.

Return ONLY:
{"bullets": ["5 bullets max, each one sentence"],
 "players_discussed": ["names as written"],
 "explicit_picks": [{"player": "...", "call": "buy|sell|captain|avoid|start|bench"}]}

Only list explicit picks the presenter actually made. Do not infer picks from tone.
"""

EXPLAIN_SYSTEM = """You explain an FPL squad recommendation to its owner.

You receive the optimiser's plan, the model's numbers, and supporting evidence. Write
2-4 short paragraphs of plain British English.

Rules:
- Every number you state must come from the data given. Never invent a statistic.
- Name the single biggest reason first.
- Say plainly when the case is thin or the model is uncertain. Do not oversell.
- If the recommendation is to do nothing, explain why that is the right call.
Return plain prose, not JSON.
"""

CRITIQUE_SYSTEM = """You are an adversarial reviewer of an FPL plan. Argue against it.

You receive the plan, the model's numbers, the strongest opposing claims from the text
corpus, and the last four weeks of this manager's accepted/rejected recommendations.

Return ONLY:
{"concerns": [{"player": "...", "issue": "...", "severity": "low|medium|high",
               "evidence_ids": [int]}],
 "overlooked_alternatives": [{"swap": "X -> Y", "argument": "...", "est_delta": float}],
 "confidence_in_plan": 0.0-1.0,
 "one_line_verdict": "..."}

You cannot change the plan. Any alternative you propose is re-solved by the optimiser
and shown with its real computed cost, so guessing wildly will simply be exposed.
"""

CHAT_SYSTEM = """You are an FPL analyst assistant with read-only tools over this app's data.

Rules that matter:
- Every numeric claim must come from a tool result. Never invent statistics.
- Cite prediction ids when you quote projections.
- You may propose squad changes, but you cannot apply them — say so, and the UI will
  offer the user an "apply to planned squad" button.
- If a tool returns nothing, say so rather than filling the gap with a guess.
- British English. Concise. No hype.
"""

DIGEST_SYSTEM = """Write a short weekly FPL digest for one manager, in British English.

Cover: the recommended action and why, what changed this week among their players, any
chip-expiry warning, and one thing worth watching. Under 250 words. No hype, no emojis
beyond the ones supplied in the structure.
"""

SETTINGS_ASSISTANT_SYSTEM = """You translate a plain-English request into FPL squad settings.

Return ONLY a JSON object with any subset of these keys:
  risk (-1..1), horizon_gws (1..8), horizon_decay (0.5..1.0), bench_weight (0..0.5),
  max_hits_per_gw (0..3), min_expected_gain_to_act (0..5), prefer_differentials (bool),
  rank_mode ("maximise_points"|"climb_rank"|"protect_rank"), notes (string)

Chasing from behind implies higher risk, differentials and climb_rank. Defending a lead
implies lower risk and protect_rank. Only include keys the request actually justifies.
"""


TASKS: dict[str, LLMTask] = {
    "extract_claims": LLMTask(
        name="extract_claims", temperature=0.1, max_tokens=1500,
        system=EXTRACT_CLAIMS_SYSTEM, response_schema=dict, cacheable=True,
    ),
    "resolve_entity": LLMTask(
        name="resolve_entity", temperature=0.0, max_tokens=200,
        system=RESOLVE_ENTITY_SYSTEM, response_schema=dict, cacheable=True,
    ),
    "classify_injury_severity": LLMTask(
        name="classify_injury_severity", temperature=0.1, max_tokens=300,
        system=INJURY_SEVERITY_SYSTEM, response_schema=dict, cacheable=True,
    ),
    "dedupe_claims": LLMTask(
        name="dedupe_claims", temperature=0.0, max_tokens=500,
        system="Return {\"same\": true|false} for whether two claims assert the same fact.",
        response_schema=dict, cacheable=True,
    ),
    "summarise_video": LLMTask(
        name="summarise_video", temperature=0.3, max_tokens=800,
        system=SUMMARISE_VIDEO_SYSTEM, response_schema=dict, cacheable=True,
    ),
    "summarise_player_week": LLMTask(
        name="summarise_player_week", temperature=0.3, max_tokens=600,
        system="Summarise what was said about this player this week, citing sources. "
               "Plain prose, under 150 words.",
        json_only=False, cacheable=True,
    ),
    "explain_recommendation": LLMTask(
        name="explain_recommendation", temperature=0.4, max_tokens=900,
        system=EXPLAIN_SYSTEM, json_only=False, cacheable=False,
    ),
    "critique_recommendation": LLMTask(
        name="critique_recommendation", temperature=0.5, max_tokens=1200,
        system=CRITIQUE_SYSTEM, response_schema=dict, cacheable=False,
    ),
    "chat": LLMTask(
        name="chat", temperature=0.6, max_tokens=2000,
        system=CHAT_SYSTEM, json_only=False, cacheable=False,
    ),
    "weekly_digest": LLMTask(
        name="weekly_digest", temperature=0.5, max_tokens=700,
        system=DIGEST_SYSTEM, json_only=False, cacheable=False,
    ),
    "settings_assistant": LLMTask(
        name="settings_assistant", temperature=0.2, max_tokens=500,
        system=SETTINGS_ASSISTANT_SYSTEM, response_schema=dict, cacheable=False,
    ),
}


def get(name: str) -> LLMTask:
    if name not in TASKS:
        raise KeyError(f"unknown LLM task {name!r}")
    return TASKS[name]
