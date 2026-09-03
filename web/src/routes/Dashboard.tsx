/** The one screen you'd keep if you could only have one. docs/10 screen 1. */
import { useState } from "react";
import {
  money,
  pts,
  useDeadline,
  useExplain,
  useFeed,
  useHealth,
  useRecommend,
  useRecommendations,
  type Recommendation,
} from "../lib/api";
import { useSquadStore } from "../stores/squad";
import { Card, Delta, DistributionBar, Empty, ErrorNote, Loading, PlayerName, PlayerRow, PositionBadge } from "../components/ui";

const VARIANTS = ["safe", "balanced", "aggressive"] as const;

const CHIP_LABEL: Record<string, string> = {
  wildcard: "Wildcard",
  freehit: "Free Hit",
  bboost: "Bench Boost",
  "3xc": "Triple Captain",
};

/** Loud when the app wants a chip played THIS week; quiet line for a later-horizon
 * plan. A chip suggestion must be impossible to miss — it changes the whole week. */
function ChipCallout({ payload }: { payload: Recommendation["payload"] }) {
  const cal = payload.chip_calendar?.find((c) => c.chip === payload.chip);
  const label = CHIP_LABEL[payload.chip ?? ""] ?? payload.chip;
  const later = payload.future_gameweeks?.filter((g) => g.chip) ?? [];
  if (!payload.chip && later.length === 0) return null;
  return (
    <div className="space-y-2 mt-3">
      {payload.chip && (
        <div className="rounded-lg border border-pos/40 bg-pos/10 px-4 py-3">
          <p className="font-semibold text-pos text-base">
            ▶ Play {label} in GW{payload.gameweek}
          </p>
          {(cal?.reason || cal?.actionable) && (
            <p className="text-sm text-muted mt-1">
              {cal?.reason}
              {cal?.actionable ? ` · calendar: best spot GW${cal.best_gw} (+${cal.gain})` : ""}
            </p>
          )}
        </div>
      )}
      {later.map((g) => (
        <div key={g.gameweek} className="rounded-lg border border-line px-4 py-2.5">
          <p className="text-sm text-slate-300">
            <span className="text-warn font-medium">{CHIP_LABEL[g.chip ?? ""] ?? g.chip}</span>{" "}
            planned for GW{g.gameweek}
          </p>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const { activeSquadId, variant, setVariant } = useSquadStore();
  const { data: deadline } = useDeadline();
  const { data: recs, isLoading, error } = useRecommendations(activeSquadId ?? undefined);
  const recommend = useRecommend(activeSquadId ?? undefined);
  const explain = useExplain();
  const { data: health } = useHealth();
  const [rationale, setRationale] = useState<string | null>(null);

  const rec = recs?.find((r) => r.variant === variant) ?? recs?.[0];

  if (!activeSquadId) {
    return <Empty title="No squad yet" hint="Create one from the squad picker to get started." />;
  }

  return (
    <div className="space-y-4 max-w-5xl mx-auto">
      <ErrorNote error={error} />

      {/* Headline: the recommended action in one sentence. */}
      <Card
        title="This gameweek"
        actions={
          <button
            className="btn btn-primary"
            onClick={() => recommend.mutate({ force_refresh: true })}
            disabled={recommend.isPending}
          >
            {recommend.isPending ? "Solving…" : "Recommend now"}
          </button>
        }
      >
        {isLoading ? (
          <Loading label="Loading recommendation" />
        ) : !rec ? (
          <Empty
            title="No recommendation yet"
            hint="Run 'Recommend now' — it needs predictions for this gameweek first."
          />
        ) : (
          <>
            <div className="flex flex-wrap gap-2 mb-3">
              {VARIANTS.map((v) => (
                <button
                  key={v}
                  onClick={() => setVariant(v)}
                  className={`chip capitalize ${
                    v === variant ? "bg-pos/15 border-pos/40 text-pos" : "text-muted"
                  }`}
                  disabled={!recs?.some((r) => r.variant === v)}
                >
                  {v}
                </button>
              ))}
            </div>

            <p className="text-lg leading-snug">{rec.payload.headline}</p>

            <ChipCallout payload={rec.payload} />

            {rec.payload.recommendation === "do_nothing" && (
              <p className="mt-2 text-sm text-warn">
                Doing nothing is a real recommendation, not a failure to find one.
              </p>
            )}

            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
              <Stat label="Projected" value={pts(rec.payload.totals.exp_points_gw)} />
              <Stat label="Spread (SD)" value={`±${pts(rec.payload.totals.sd_points_gw)}`} />
              <Stat
                label={`Horizon (${rec.payload.horizon?.gws ?? 5} GW)`}
                value={pts(rec.payload.totals.exp_points_horizon)}
              />
              <Stat label="In the bank" value={money(rec.payload.totals.bank_after)} />
            </dl>

            {rec.payload.transfers.length > 0 && (
              <ul className="mt-4 space-y-2">
                {rec.payload.transfers.map((t, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm flex-wrap">
                    <span className="text-neg">
                      <PlayerName player={t.out} />
                    </span>
                    <span className="text-muted">→</span>
                    <span className="text-pos">
                      <PlayerName player={t.in} />
                    </span>
                    <PositionBadge position={t.in.position} />
                    <span className="text-muted num">{money(t.in.price)}</span>
                    <Delta value={t.delta_exp_points_gw} />
                  </li>
                ))}
              </ul>
            )}

            <div className="flex flex-wrap gap-2 mt-4 text-sm">
              {rec.payload.alternatives.map((a, i) => (
                <span key={i} className="chip text-muted">
                  {a.label} <Delta value={a.delta} />
                </span>
              ))}
            </div>

            {rec.id && (
              <button
                className="btn mt-4"
                onClick={() =>
                  explain.mutate(rec.id!, {
                    onSuccess: (d) =>
                      setRationale(d.rationale ?? "No LLM configured — set LLM_API_KEY to enable."),
                  })
                }
                disabled={explain.isPending}
              >
                {explain.isPending ? "Thinking…" : "Why?"}
              </button>
            )}
            {rationale && (
              <p className="mt-3 text-sm text-slate-300 whitespace-pre-wrap border-l-2 border-line pl-3">
                {rationale}
              </p>
            )}
          </>
        )}
      </Card>

      {/* Alerts: chip expiry, flagged players, the loud stuff. */}
      {rec && rec.payload.chip_warnings?.length > 0 && (
        <Card title="Alerts">
          <ul className="space-y-2">
            {rec.payload.chip_warnings.map((w, i) => (
              <li
                key={i}
                className={`text-sm ${w.severity === "critical" ? "text-neg" : "text-warn"}`}
              >
                {w.message}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Projected XI with the P10–P90 band drawn per player. */}
      {rec && (
        <Card title={`Projected XI · ${rec.payload.lineup.formation}`}>
          <ul className="divide-y divide-line">
            {rec.payload.lineup.xi.map((p) => (
              <PlayerRow
                key={p.player_id}
                player={p}
                badges={
                  <>
                    {p.player_id === rec.payload.lineup.captain && (
                      <span className="ml-1.5 chip bg-pos/15 text-pos border-pos/40">C</span>
                    )}
                    {p.player_id === rec.payload.lineup.vice && (
                      <span className="ml-1.5 chip text-muted">V</span>
                    )}
                  </>
                }
                meter={
                  <span className="hidden sm:block w-28">
                    <DistributionBar
                      p10={Math.max(0, (p.exp_points ?? 0) - 3)}
                      p90={(p.exp_points ?? 0) + 5}
                      mean={p.exp_points ?? 0}
                    />
                  </span>
                }
              />
            ))}
          </ul>
          <p className="text-xs text-muted mt-3">
            Bench: {rec.payload.lineup.bench_order.map((p) => p.name).join(", ") || "—"}
          </p>
        </Card>
      )}

      {rec && rec.payload.chip_calendar?.length > 0 && (
        <Card title="Chip calendar">
          <ul className="space-y-1.5 text-sm">
            {rec.payload.chip_calendar.map((c) => (
              <li key={c.chip} className="flex justify-between gap-3">
                <span>{c.label}</span>
                <span className="text-muted flex-1 truncate">{c.reason}</span>
                {/* Naming a gameweek before doubles and blanks exist is false precision. */}
                {c.actionable ? (
                  <span className="num">
                    GW{c.best_gw} <span className="text-pos">+{c.gain}</span>
                  </span>
                ) : (
                  <span className="text-muted text-xs whitespace-nowrap">not yet</span>
                )}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <WhatChanged />

      {health && (
        <p className="text-xs text-muted">
          {health.players} players · {health.predictions} predictions ·{" "}
          {health.connectors_available}/{health.connectors} sources available ·{" "}
          {health.llm_configured ? "LLM configured" : "no LLM key"} · season {health.season}
          {deadline?.chips_available?.length ? ` · chips: ${deadline.chips_available.join(", ")}` : ""}
        </p>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-muted">{label}</dt>
      <dd className="num text-xl">{value}</dd>
    </div>
  );
}

/** "What changed since you last looked" — new claims, newest first. */
function WhatChanged() {
  const { data } = useFeed("claim");
  const items = (data ?? []).slice(0, 6);
  if (!items.length) return null;
  return (
    <Card title="What changed">
      <ul className="space-y-2">
        {items.map((c) => (
          <li key={`${c.type}-${c.id}`} className="text-sm feed">
            <span className="text-muted">{c.claim_type}</span>{" "}
            {c.player_name && <span className="font-medium">{c.player_name}</span>}{" "}
            <span className="text-slate-300">“{c.text_span}”</span>{" "}
            <span className="text-muted text-xs">{c.source_id}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

export { Stat };
