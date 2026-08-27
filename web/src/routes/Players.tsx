/** Player explorer + detail. docs/10 screens 4 and 5. */
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  money,
  pct,
  pts,
  usePlayer,
  usePlayerClaims,
  usePlayerFeatures,
  usePlayerHistory,
  usePlayers,
  type PlayerRow,
} from "../lib/api";
import { useSquadStore } from "../stores/squad";
import { Card, Empty, FixturePips, Loading, PlayerName, PositionBadge, Sparkline, StatusDot } from "../components/ui";

const COLUMNS: { key: keyof PlayerRow; label: string; fmt?: (v: any) => string; hideSm?: boolean }[] = [
  { key: "price", label: "£", fmt: money },
  { key: "exp_points_gw", label: "EP", fmt: (v) => pts(v) },
  { key: "exp_points_horizon", label: "EP5", fmt: (v) => pts(v), hideSm: true },
  { key: "value", label: "EP/£m", fmt: (v) => pts(v, 2), hideSm: true },
  { key: "p_start", label: "P(start)", fmt: (v) => pct(v), hideSm: true },
  { key: "p_haul", label: "P(haul)", fmt: (v) => pct(v, 1), hideSm: true },
  { key: "owned_pct", label: "Own%", fmt: (v) => (v == null ? "—" : `${v}%`), hideSm: true },
];

export function PlayerExplorer() {
  const [position, setPosition] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [minMinutes, setMinMinutes] = useState("");
  const [sort, setSort] = useState<keyof PlayerRow>("exp_points_gw");
  const [term, setTerm] = useState("");
  const { watchlist, toggleWatch } = useSquadStore();

  const { data, isLoading } = usePlayers({
    position: position || undefined,
    max_price: maxPrice ? Number(maxPrice) * 10 : undefined,
    min_minutes: minMinutes || undefined,
    sort,
    limit: 600,
  });

  const rows = useMemo(
    () =>
      (data ?? []).filter((p) =>
        !term ? true : p.web_name.toLowerCase().includes(term.toLowerCase()),
      ),
    [data, term],
  );

  return (
    <div className="space-y-4">
      <Card title="Filters">
        <div className="flex flex-wrap gap-2">
          <input
            className="bg-raised border border-line rounded-lg px-3 py-2 text-sm flex-1 min-w-[10rem]"
            placeholder="Search name"
            value={term}
            onChange={(e) => setTerm(e.target.value)}
          />
          <select
            className="bg-raised border border-line rounded-lg px-3 py-2 text-sm"
            value={position}
            onChange={(e) => setPosition(e.target.value)}
          >
            <option value="">All positions</option>
            {["GK", "DEF", "MID", "FWD"].map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
          <input
            className="bg-raised border border-line rounded-lg px-3 py-2 text-sm w-28 num"
            placeholder="max £m"
            value={maxPrice}
            onChange={(e) => setMaxPrice(e.target.value)}
          />
          <input
            className="bg-raised border border-line rounded-lg px-3 py-2 text-sm w-32 num"
            placeholder="min minutes"
            value={minMinutes}
            onChange={(e) => setMinMinutes(e.target.value)}
          />
        </div>
      </Card>

      {isLoading ? (
        <Loading label="Loading players" />
      ) : (
        <Card title={`${rows.length} players`}>
          <div className="scroll-x">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-muted text-xs uppercase tracking-wide">
                  <th className="text-left py-2 pr-2">Player</th>
                  {COLUMNS.map((c) => (
                    <th
                      key={String(c.key)}
                      className={`text-right py-2 px-2 cursor-pointer whitespace-nowrap ${
                        c.hideSm ? "hidden sm:table-cell" : ""
                      } ${sort === c.key ? "text-slate-100" : ""}`}
                      onClick={() => setSort(c.key)}
                    >
                      {c.label}
                    </th>
                  ))}
                  <th className="text-right py-2 pl-2 hidden sm:table-cell">Form</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 300).map((p) => (
                  <tr key={p.id} className="border-t border-line hover:bg-raised/60">
                    <td className="py-2 pr-2">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => toggleWatch(p.id)}
                          className={watchlist.includes(p.id) ? "text-warn" : "text-muted/40"}
                          aria-label="Watchlist"
                        >
                          ★
                        </button>
                        <PositionBadge position={p.position} />
                        <Link to={`/players/${p.id}`} className="hover:underline truncate">
                          <PlayerName player={p} />
                        </Link>
                      </div>
                    </td>
                    {COLUMNS.map((c) => (
                      <td
                        key={String(c.key)}
                        className={`text-right num py-2 px-2 ${c.hideSm ? "hidden sm:table-cell" : ""}`}
                      >
                        {c.fmt ? c.fmt(p[c.key]) : String(p[c.key] ?? "—")}
                      </td>
                    ))}
                    <td className="text-right py-2 pl-2 hidden sm:table-cell">
                      <Sparkline values={p.form_sparkline ?? []} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

const TABS = ["Projection", "Form", "News & video", "Fixtures"] as const;

export function PlayerDetail() {
  const { id } = useParams();
  const playerId = Number(id);
  const [tab, setTab] = useState<(typeof TABS)[number]>("Projection");
  const { data: player, isLoading } = usePlayer(playerId);
  const { data: features } = usePlayerFeatures(playerId);
  const { data: history } = usePlayerHistory(playerId);
  const { data: claims } = usePlayerClaims(playerId);

  if (isLoading) return <Loading />;
  if (!player) return <Empty title="Player not found" />;

  const pred = player.prediction;

  return (
    <div className="space-y-4 max-w-4xl">
      <Card>
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h1 className="text-2xl font-semibold">{player.web_name}</h1>
            <p className="text-muted text-sm flex items-center gap-2">
              <PositionBadge position={player.position} />
              {player.team_name} · {money(player.price?.price)}
              {player.availability?.[0] && (
                <StatusDot
                  status={player.availability[0].status}
                  title={player.availability[0].note || player.availability[0].status}
                />
              )}
            </p>
          </div>
          {pred && (
            <div className="text-right">
              <div className="num text-3xl">{pts(pred.exp_points)}</div>
              <div className="text-xs text-muted">
                ±{pts(pred.sd_points)} · P(haul) {pct(pred.p_haul_10, 1)}
              </div>
            </div>
          )}
        </div>
      </Card>

      <div className="flex gap-2 flex-wrap">
        {TABS.map((t) => (
          <button
            key={t}
            className={`chip ${t === tab ? "bg-pos/15 border-pos/40 text-pos" : "text-muted"}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "Projection" && (
        <Card title="Component breakdown">
          {!pred ? (
            <Empty title="No projection yet" hint="Run a predict pass for this gameweek." />
          ) : (
            <>
              {/* The waterfall people actually want: how the number was assembled. */}
              <ul className="space-y-1.5 text-sm">
                <Row label="P(start)" value={pct(pred.p_start)} />
                <Row label="Expected minutes" value={pts(pred.exp_minutes, 0)} />
                <Row label="Expected goals" value={pts(pred.exp_goals, 2)} />
                <Row label="Expected assists" value={pts(pred.exp_assists, 2)} />
                <Row label="P(clean sheet)" value={pct(pred.p_clean_sheet)} />
                <Row label="DefCon points" value={pts(pred.exp_defcon_points, 2)} />
                <Row label="Bonus" value={pts(pred.exp_bonus, 2)} />
                {pred.exp_saves ? <Row label="Saves" value={pts(pred.exp_saves, 2)} /> : null}
                <li className="flex justify-between border-t border-line pt-2 font-medium">
                  <span>Expected points</span>
                  <span className="num">{pts(pred.exp_points, 2)}</span>
                </li>
              </ul>

              {/* Adjustments are shown separately, never folded into the model number. */}
              {pred.adjustment !== 0 && (
                <p className="mt-3 text-sm text-warn">
                  model {pts(pred.base_exp_points, 2)} → adjusted {pts(pred.exp_points, 2)} —{" "}
                  {pred.adjustment_reason}
                </p>
              )}

              {features && features.length > 0 && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm text-muted">
                    {features.length} features behind this number
                  </summary>
                  <ul className="mt-2 grid sm:grid-cols-2 gap-x-6 text-xs">
                    {features.map((f: any) => (
                      <li key={f.name} className="flex justify-between py-0.5" title={f.description}>
                        <span className="text-muted truncate">{f.name}</span>
                        <span className="num">
                          {f.value?.toFixed?.(2) ?? f.value}
                          {f.percentile != null && (
                            <span className="text-muted"> ({Math.round(f.percentile * 100)}%)</span>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </Card>
      )}

      {tab === "Form" && (
        <Card title="Per-fixture history">
          <div className="scroll-x">
            <table className="w-full text-sm">
              <thead className="text-muted text-xs uppercase">
                <tr>
                  <th className="text-left py-1">GW</th>
                  <th className="text-left py-1">Opp</th>
                  <th className="text-right py-1">Min</th>
                  <th className="text-right py-1">G</th>
                  <th className="text-right py-1">A</th>
                  <th className="text-right py-1">xG</th>
                  <th className="text-right py-1">xA</th>
                  <th className="text-right py-1">BPS</th>
                  <th className="text-right py-1">Pts</th>
                </tr>
              </thead>
              <tbody>
                {(history ?? []).slice(0, 40).map((h: any) => (
                  <tr key={h.id} className="border-t border-line">
                    <td className="py-1">{h.gameweek}</td>
                    <td className="py-1 truncate">{h.opponent}</td>
                    <td className="text-right num py-1">{h.minutes}</td>
                    <td className="text-right num py-1">{h.goals_scored}</td>
                    <td className="text-right num py-1">{h.assists}</td>
                    <td className="text-right num py-1">{h.xg?.toFixed?.(2) ?? "—"}</td>
                    <td className="text-right num py-1">{h.xa?.toFixed?.(2) ?? "—"}</td>
                    <td className="text-right num py-1">{h.bps ?? "—"}</td>
                    <td className="text-right num py-1 font-medium">{h.total_points}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === "News & video" && (
        <Card title="What's being said">
          {!claims?.length ? (
            <Empty
              title="Nothing extracted yet"
              hint="The text pipeline needs an LLM key and an ingest run."
            />
          ) : (
            <ul className="space-y-3">
              {claims.map((c: any) => (
                <li key={c.id} className="text-sm feed">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="chip text-muted">{c.claim_type}</span>
                    {c.stance && (
                      <span className={c.stance === "positive" ? "text-pos" : "text-neg"}>
                        {c.stance}
                      </span>
                    )}
                    <span className="text-muted text-xs">{c.source_id}</span>
                    {c.duplicate_count > 1 && (
                      <span className="chip text-muted">seen on {c.duplicate_count} sites</span>
                    )}
                  </div>
                  <p className="text-slate-300 mt-0.5">“{c.text_span}”</p>
                  {c.deep_link && (
                    <a
                      href={c.deep_link}
                      target="_blank"
                      rel="noreferrer"
                      className="text-xs text-pos hover:underline"
                    >
                      {c.start_s != null ? `jump to ${Math.floor(c.start_s)}s` : "source"}
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}

      {tab === "Fixtures" && (
        <Card title="Next 8">
          <FixturePips fixtures={player.upcoming ?? []} />
          <ul className="mt-3 space-y-1 text-sm">
            {(player.upcoming ?? []).map((f: any, i: number) => (
              <li key={i} className="flex justify-between">
                <span>
                  GW{f.gameweek} {f.opponent} ({f.is_home ? "H" : "A"})
                </span>
                <span className="num text-muted">
                  {f.competition !== "PL" ? f.competition : `FDR ${f.difficulty ?? "?"}`}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <li className="flex justify-between">
      <span className="text-muted">{label}</span>
      <span className="num">{value}</span>
    </li>
  );
}
