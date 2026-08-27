# 08 — LLM Layer

One client, OpenAI-compatible (`base_url` + `api_key` from `.env`), so OpenRouter, NVIDIA build, or a local Ollama all work identically. Per-task model selection lives in **global settings** and is editable in the UI.

```python
class LLMTask(BaseModel):
    name: str
    model: str                  # resolved: settings["llm.tasks"][name] or LLM_DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 2000
    response_schema: type[BaseModel] | None
    cacheable: bool = True
```

Every call is logged to `llm_calls` with a `prompt_hash`; cacheable tasks return the cached row on an exact hit. Structured tasks use JSON-schema-constrained output where the provider supports it, and fall back to "reply with JSON only" plus a repair retry (one attempt, then mark the item failed and move on — never loop).

## Task registry

| Task | Suggested model class | Volume | Purpose |
|---|---|---|---|
| `extract_claims` | cheap/fast (e.g. a small hosted model, or local Qwen/Llama on the 3070) | very high — every chunk | The workhorse. Turns text into `claims` rows |
| `resolve_entity` | cheap | high, heavily cached | Nickname → player_id with club context |
| `classify_injury_severity` | mid | medium | "Knock, assessed daily" → status + expected GWs out + confidence |
| `dedupe_claims` | cheap embeddings + mid model for edge cases | medium | Collapse semantic duplicates |
| `summarise_video` | mid | one per video | 5-bullet summary + list of players discussed + explicit picks |
| `summarise_player_week` | mid | per watchlist player per day | One paragraph of "what's being said", with citations |
| `explain_recommendation` | **strong** | a few per day | The headline rationale for a squad/transfer plan |
| `critique_recommendation` | **strong** | a few per day | Adversarial pass — see below |
| `chat` | **strong** | interactive | The Q&A interface, tool-enabled |
| `weekly_digest` | mid | weekly | Discord narrative |
| `settings_assistant` | mid | rare | "Make me a squad tuned for a 12-person work league where I'm 40 points behind" → proposed settings JSON |

Sensible defaults to seed: `extract_claims` and `resolve_entity` → a cheap fast model or local Ollama; `explain_recommendation`, `critique_recommendation`, `chat` → your strongest available. All overridable per task in Settings → Global → LLM.

## Claim extraction prompt (the important one)

Input: one chunk (≈800 tokens) plus metadata (source, channel, published date, current gameweek) plus a candidate player list (players mentioned by fuzzy match in this document, with clubs).

Output: array of claims conforming to the `claims` schema — `player`, `claim_type`, `stance`, `sentiment`, `confidence`, `horizon_gw`, `text_span`.

Rules baked into the system prompt:
- Extract only what is *asserted in this chunk*. No inference from world knowledge.
- `text_span` must be a short verbatim fragment from the chunk (kept under 15 words) so the evidence panel can anchor and deep-link it. Never reproduce whole paragraphs of a source into the DB.
- If the player is ambiguous, return `player: null` with the surface form — the resolver handles it.
- Distinguish the speaker's *own view* from them *reporting others'* ("Ornstein says…" is a report, not the pundit's opinion) — set `is_reported`.
- Return `[]` freely. Most chunks contain no claims, and a model that always finds something is worse than useless.

Few-shot examples in the prompt should include at least one "banter, no claim" case, because FPL YouTube is 60% banter.

## The adjustment layer — and its leash

Text signals reach predictions through two routes, and only two:

1. **As features** (`05` section F), learned by the model like anything else. This is the preferred route.
2. **As a bounded post-hoc adjustment**, for information that is genuinely too new to be in any feature — the classic case being breaking injury news 30 minutes before a deadline.

The adjustment is hard-capped: `|adjustment| ≤ min(2.0 points, 0.25 × base_exp_points)`, it must cite at least two independent claims (near-dupe-collapsed) or one tier-1 source, it writes `adjustment_reason` and evidence links, and it is shown separately in the UI as "model 5.8 → adjusted 4.9 (why)". You can disable the whole layer with one toggle.

The one exception that may exceed the cap: an availability override (suspension, confirmed non-travel, confirmed lineup omission) routes through the minutes model as a hard fact, not through the adjustment layer.

## Critique pass

After the optimiser produces its variants, a strong model gets: the plan, the model's numbers, the top opposing claims from the text corpus, and the last four weeks of this squad's accepted/rejected recommendations. It returns structured critique:

```json
{
  "concerns": [{"player": "...", "issue": "rotation risk before UCL", "severity": "medium", "evidence_ids": [123, 456]}],
  "overlooked_alternatives": [{"swap": "X → Y", "argument": "...", "est_delta": -0.4}],
  "confidence_in_plan": 0.72,
  "one_line_verdict": "..."
}
```

It cannot mutate the plan. If it proposes an alternative, that alternative is fed back to the optimiser as a *constrained re-solve* (force the swap, re-optimise the rest) and shown with its real model-computed delta — so you see the honest cost of the LLM's idea rather than the LLM's guess at it. That single design choice is what stops this becoming a vibes machine.

## Chat

Tool-calling agent with read-only tools: `search_corpus(query, filters)` (hybrid sqlite-vec + FTS5), `get_player(id)`, `get_prediction(player, gw)`, `compare_players([...])`, `run_optimiser(constraints)`, `explain_feature(player, feature)`, `get_squad_state(squad_id)`, `list_fixtures(team, n)`.

`run_optimiser` is what makes iterative team-building work: "what if I go without a premium striker?" becomes a constrained re-solve, answered with real numbers. The chat is scoped to the active squad and can be pointed at a comparison set ("compare my two squads for GW14").

Guardrails: the chat may propose but never persist changes — applying a plan is always a UI action by you. Every numeric claim in a chat answer must come from a tool result; the system prompt forbids inventing statistics, and answers cite prediction ids.

## Cost control

You said no in-app limit needed, but the plumbing is free: the `llm_calls` table gives per-task daily cost in the UI, so if `extract_claims` is eating £4/day you'll see it and can switch that task to Ollama in one dropdown. Chunk-level caching means a re-run of extraction over unchanged text costs nothing.
