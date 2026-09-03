"""The adjustment layer — and its leash. docs/08.

Text signals reach predictions through exactly two routes: as learned features (the
preferred one), and as this bounded post-hoc adjustment, for information genuinely too
new to be in any feature — breaking injury news 30 minutes before a deadline.

The leash, enforced here rather than trusted to a prompt:
  |adjustment| <= min(2.0 points, 0.25 x base_exp_points)
  must cite >= 2 independent near-dupe-collapsed claims, or 1 tier-1 source
  writes adjustment_reason and evidence links
  the whole layer is one settings toggle away from off

The one exception that may exceed the cap is an availability override — suspension,
confirmed non-travel, confirmed lineup omission. Those are *facts*, so they route
through the minutes model instead, not through here.
"""

from __future__ import annotations

import logging

from ..db.engine import query, utcnow, writer
from ..db.settings_store import global_settings

log = logging.getLogger(__name__)

TIER1_TRUST = 1.5
RECENT_HOURS = 36

# Claim types this layer will act on. The extractor's vocabulary is wider than this on
# purpose — the test is whether a claim carries *availability or role* information that no
# feature can already see:
#   in   injury / return / rotation / role_change / penalty_duty — availability and role
#   in   manager_quote — the pre-deadline press conference, which is exactly the "too new
#        for any feature" case this layer exists for, and was the single biggest omission:
#        it is 43 of the 267 claims held, and adding it takes the layer from 2 players it
#        could ever fire on to 6
#   out  form — already modelled properly by xg90/xa90/bps90; adjusting on it double-counts
#   out  recommendation / captain_pick / avoid — pundit opinion, not information. Beating
#        those is the point of the model; feeding them back in is circular
#   out  transfer_rumour — mostly about players who then stay, and the deadline-day cases
#        arrive as `rotation` anyway
ADJUSTABLE_CLAIM_TYPES = (
    "injury", "return", "rotation", "role_change", "penalty_duty", "manager_quote",
)

# Fallback direction when the extractor returned no sentiment, for the types whose name
# already carries one. Everything else abstains rather than guessing.
DEFAULT_SENTIMENT = {"injury": -0.5, "rotation": -0.5, "return": 0.3, "penalty_duty": 0.3}


def _cap(base: float, settings: dict) -> float:
    max_abs = float(settings.get("adjustment.max_points", 2.0))
    fraction = float(settings.get("adjustment.max_fraction", 0.25))
    return min(max_abs, fraction * abs(base))


def recent_claims(player_id: int, since_hours: int = RECENT_HOURS) -> list[dict]:
    """Near-dupe-collapsed: one row per semantic group, so six syndicated copies of one
    story count once."""
    placeholders = ",".join("?" * len(ADJUSTABLE_CLAIM_TYPES))
    rows = query(
        # Trust comes from the channel for a video and from the source otherwise. The
        # channel join alone left every RSS, Reddit and Bluesky claim on a NULL weight,
        # so `TIER1_TRUST` was unreachable for anything that was not a YouTube video —
        # and `sources.trust_weight` is the column the admin UI already edits.
        "SELECT c.*, COALESCE(ch.trust_weight, s.trust_weight) trust_weight, rd.source_id, "
        "COALESCE(c.semantic_group, c.id) grp "
        "FROM claims c JOIN raw_documents rd ON rd.id=c.raw_doc_id "
        "LEFT JOIN sources s ON s.id=rd.source_id "
        "LEFT JOIN videos v ON v.raw_doc_id=c.raw_doc_id "
        "LEFT JOIN channels ch ON ch.channel_id=v.channel_id "
        "WHERE c.player_id=? AND c.extracted_at > datetime('now', ?) "
        f"AND c.claim_type IN ({placeholders}) "
        "ORDER BY c.extracted_at DESC",
        (player_id, f"-{since_hours} hour", *ADJUSTABLE_CLAIM_TYPES),
    )
    seen: set[int] = set()
    out = []
    for r in rows:
        if r["grp"] in seen:
            continue
        seen.add(r["grp"])
        out.append(dict(r))
    return out


def compute_adjustment(player_id: int, base_exp_points: float) -> tuple[float, str | None, list[int]]:
    """(adjustment, reason, claim_ids). Returns (0, None, []) unless the evidence bar is met."""
    settings = global_settings()
    if not settings.get("adjustment.enabled", True):
        return 0.0, None, []

    claims = recent_claims(player_id)
    if not claims:
        return 0.0, None, []

    trust_weights = settings.get("text.trust_weights", {})
    tier1 = [c for c in claims if float(c.get("trust_weight") or 0) >= TIER1_TRUST]
    independent = len({c["grp"] for c in claims})

    # The evidence bar: two independent collapsed claims, or one tier-1 source.
    if independent < 2 and not tier1:
        return 0.0, None, []

    signal = 0.0
    total_weight = 0.0
    for c in claims:
        w = float(c.get("trust_weight") or trust_weights.get("rss_default", 1.0))
        w *= float(c.get("confidence") or 0.5)
        if c.get("is_reported"):
            w *= 0.8
        sentiment = c.get("sentiment")
        if sentiment is None:
            # Only a type whose *name* already states a direction gets a default. A manager
            # quote or a role change with no extracted sentiment says nothing either way,
            # and the old blanket +0.3 manufactured a positive signal out of silence.
            sentiment = DEFAULT_SENTIMENT.get(c["claim_type"])
            if sentiment is None:
                continue
        signal += w * float(sentiment)
        total_weight += w

    if total_weight <= 0:
        return 0.0, None, []

    raw = (signal / total_weight) * abs(base_exp_points) * 0.35
    cap = _cap(base_exp_points, settings)
    adjustment = max(-cap, min(cap, raw))
    if abs(adjustment) < 0.05:
        return 0.0, None, []

    direction = "down" if adjustment < 0 else "up"
    top = max(claims, key=lambda c: float(c.get("trust_weight") or 0))
    reason = (
        f"Adjusted {direction} {abs(adjustment):.2f} pts on {independent} independent "
        f"claim{'s' if independent != 1 else ''} in the last {RECENT_HOURS}h "
        f'(e.g. {top["source_id"]}: "{top["text_span"]}")'
    )
    return round(adjustment, 3), reason, [c["id"] for c in claims]


def apply_adjustments(season_id: str, gameweek: int, generated_at: str) -> int:
    """Adjust the predictions written by this run, then link the evidence."""
    rows = query(
        "SELECT id, player_id, base_exp_points, exp_points FROM predictions "
        "WHERE season_id=? AND gameweek=? AND generated_at=?",
        (season_id, gameweek, generated_at),
    )
    changed = 0
    evidence: list[tuple] = []
    with writer() as conn:
        for r in rows:
            base = r["base_exp_points"] if r["base_exp_points"] is not None else r["exp_points"]
            adj, reason, claim_ids = compute_adjustment(r["player_id"], base)
            if adj == 0.0:
                continue
            conn.execute(
                "UPDATE predictions SET exp_points=?, adjustment=?, adjustment_reason=? WHERE id=?",
                (max(0.0, base + adj), adj, reason, r["id"]),
            )
            changed += 1
            for cid in claim_ids[:8]:
                evidence.append(
                    ("adjustment", r["id"], r["player_id"], "claim", cid, None, None, adj, reason)
                )
        if evidence:
            conn.executemany(
                "INSERT INTO evidence_links(subject_type,subject_id,player_id,evidence_type,"
                "claim_id,raw_doc_id,feature_name,weight,note) VALUES(?,?,?,?,?,?,?,?,?)",
                evidence,
            )
    if changed:
        log.info("adjusted %d predictions for %s GW%s", changed, season_id, gameweek)
    return changed


def availability_override(player_id: int) -> dict | None:
    """The one exception to the cap — but it routes through the minutes model as a hard
    fact, not through the adjustment layer. Returned here so callers can see it."""
    rows = query(
        "SELECT status, source_id, observed_at, note FROM availability WHERE player_id=? "
        "AND status IN ('suspended','injured') AND observed_at > datetime('now','-7 day') "
        "ORDER BY observed_at DESC LIMIT 1",
        (player_id,),
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "status": r["status"],
        "source": r["source_id"],
        "observed_at": r["observed_at"],
        "note": r["note"],
        "routed_via": "minutes_model",
    }


def adjustment_report(season_id: str, gameweek: int) -> list[dict]:
    """Shown in the UI as 'model 5.8 -> adjusted 4.9 (why)'."""
    return [
        dict(r)
        for r in query(
            "SELECT p.player_id, pl.web_name name, p.base_exp_points, p.exp_points, "
            "p.adjustment, p.adjustment_reason FROM predictions p "
            "JOIN players pl ON pl.id=p.player_id "
            "WHERE p.season_id=? AND p.gameweek=? AND p.adjustment != 0 "
            "ORDER BY abs(p.adjustment) DESC",
            (season_id, gameweek),
        )
    ]


def now() -> str:
    return utcnow()
