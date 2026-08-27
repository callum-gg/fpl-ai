/** Shared primitives. Kept deliberately small — screens compose these, not a design system. */
import { type ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { money, pts } from "../lib/api";

export function Card({
  title,
  actions,
  children,
  className = "",
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card p-4 ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 mb-3">
          {typeof title === "string" ? (
            <h2 className="text-sm font-semibold text-muted uppercase tracking-wide">{title}</h2>
          ) : (
            title
          )}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Delta({ value, dp = 1 }: { value: number | null | undefined; dp?: number }) {
  if (value == null) return <span className="text-muted">—</span>;
  const tone = value > 0.05 ? "text-pos" : value < -0.05 ? "text-neg" : "text-muted";
  return (
    <span className={`num ${tone}`}>
      {value > 0 ? "+" : ""}
      {value.toFixed(dp)}
    </span>
  );
}

/** P10–P90 band with the mean marked. The distribution *is* the projection. */
export function DistributionBar({
  p10,
  p90,
  mean,
  max = 20,
}: {
  p10: number;
  p90: number;
  mean: number;
  max?: number;
}) {
  const clamp = (v: number) => Math.max(0, Math.min(100, (v / max) * 100));
  return (
    <div className="relative h-2 w-full rounded-full bg-raised" title={`P10 ${p10} · P90 ${p90}`}>
      <div
        className="absolute h-2 rounded-full bg-pos/30"
        style={{ left: `${clamp(p10)}%`, width: `${clamp(p90) - clamp(p10)}%` }}
      />
      <div
        className="absolute h-2 w-0.5 bg-pos"
        style={{ left: `${clamp(mean)}%` }}
        aria-label={`expected ${pts(mean)}`}
      />
    </div>
  );
}

export function Sparkline({ values, width = 64, height = 18 }: { values: number[]; width?: number; height?: number }) {
  if (!values?.length) return <span className="text-muted text-xs">—</span>;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const step = width / Math.max(values.length - 1, 1);
  const d = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${i * step},${height - ((v - min) / span) * height}`)
    .join(" ");
  return (
    <svg width={width} height={height} className="overflow-visible" aria-hidden="true">
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted" />
    </svg>
  );
}

const POS_TONE: Record<string, string> = {
  GK: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  DEF: "bg-sky-500/15 text-sky-300 border-sky-500/30",
  MID: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  FWD: "bg-rose-500/15 text-rose-300 border-rose-500/30",
};

export function PositionBadge({ position }: { position: string | null | undefined }) {
  if (!position) return null;
  return <span className={`chip ${POS_TONE[position] ?? ""}`}>{position}</span>;
}

/* ── players ────────────────────────────────────────────────────────────────
 * Squad picks, recommendation refs and explorer rows are three server shapes for the
 * same thing. Normalise once here so every list in the app renders players identically —
 * and so a missing name can never fall back to showing a raw id. */

export type AnyPlayer = Record<string, any>;

export function playerOf(p: AnyPlayer) {
  return {
    id: (p.player_id ?? p.id) as number,
    name: (p.name ?? p.web_name ?? null) as string | null,
    team: (p.team_short ?? null) as string | null,
    // `position` on a squad pick is its 1-15 slot number, not GK/DEF/MID/FWD.
    position: (p.position_name ?? (typeof p.position === "string" ? p.position : null)) as
      | string
      | null,
    price: (p.price ?? p.selling_price ?? p.purchase_price ?? null) as number | null,
    expPoints: (p.exp_points ?? null) as number | null,
  };
}

/** "Raya COV" — club abbreviation smaller and dimmed beside the name. */
export function PlayerName({ player }: { player: AnyPlayer }) {
  const p = playerOf(player);
  return (
    <>
      {p.name ?? `Player ${p.id}`}
      {p.team && <span className="ml-1.5 text-[0.8em] text-muted align-baseline">{p.team}</span>}
    </>
  );
}

/** One row of a player list: badge, name + club, price, EP, then whatever the screen adds. */
export function PlayerRow({
  player,
  badges,
  meter,
  children,
}: {
  player: AnyPlayer;
  badges?: ReactNode;
  meter?: ReactNode;
  children?: ReactNode;
}) {
  const p = playerOf(player);
  return (
    <li className="flex items-center gap-3 py-2">
      <PositionBadge position={p.position} />
      <span className="flex-1 truncate">
        <PlayerName player={player} />
        {badges}
      </span>
      {meter}
      {p.price != null && <span className="num text-muted">{money(p.price)}</span>}
      {p.expPoints != null && <span className="num w-12 text-right">{pts(p.expPoints)}</span>}
      {children}
    </li>
  );
}

/** Availability consensus as a single dot, with the reason in the tooltip. */
export function StatusDot({ status, title }: { status?: string | null; title?: string }) {
  const tone =
    status === "injured" || status === "suspended"
      ? "bg-neg"
      : status === "doubt"
        ? "bg-warn"
        : "bg-pos";
  return <span className={`inline-block h-2 w-2 rounded-full ${tone}`} title={title ?? status ?? ""} />;
}

export function FixturePips({
  fixtures,
}: {
  fixtures: { opponent?: string; difficulty?: number | null; is_home?: number }[];
}) {
  const tone = (d?: number | null) =>
    d == null
      ? "bg-raised"
      : d <= 2
        ? "bg-pos/60"
        : d === 3
          ? "bg-slate-500/60"
          : d === 4
            ? "bg-warn/60"
            : "bg-neg/60";
  return (
    <div className="flex gap-1">
      {fixtures.slice(0, 5).map((f, i) => (
        <span
          key={i}
          className={`h-4 w-6 rounded text-[9px] leading-4 text-center ${tone(f.difficulty)}`}
          title={`${f.opponent ?? ""} (${f.is_home ? "H" : "A"}) difficulty ${f.difficulty ?? "?"}`}
        >
          {(f.opponent ?? "").slice(0, 3)}
        </span>
      ))}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-muted text-sm py-8 justify-center">
      <span className="h-3 w-3 rounded-full border-2 border-muted border-t-transparent animate-spin" />
      {label}…
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="text-center py-10 px-4">
      <p className="text-slate-300">{title}</p>
      {hint && <p className="text-muted text-sm mt-1">{hint}</p>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="card border-neg/40 bg-neg/5 p-3 text-sm text-neg">
      {error instanceof Error ? error.message : String(error)}
    </div>
  );
}

/** Bottom sheet on mobile, centred dialog on desktop. */
export function Sheet({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    if (open) window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  // Portalled to <body>: a fixed-position sheet rendered inside an ancestor with
  // backdrop-blur/transform (e.g. the mobile header) would otherwise be contained
  // by that ancestor's box instead of the viewport.
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} aria-hidden="true" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative w-full sm:max-w-lg max-h-[85vh] overflow-y-auto card rounded-b-none sm:rounded-b-xl p-4"
      >
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold">{title}</h2>
          <button className="btn" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>,
    document.body,
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center justify-between gap-3 py-1.5 cursor-pointer">
      <span className="text-sm">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`h-5 w-9 rounded-full transition-colors ${checked ? "bg-pos/70" : "bg-raised"}`}
      >
        <span
          className={`block h-4 w-4 rounded-full bg-white transition-transform ${
            checked ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </button>
    </label>
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  label,
  format,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  label: string;
  format?: (v: number) => string;
}) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);
  return (
    <div className="py-2">
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span className="num text-muted">{format ? format(local) : local}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={local}
        onChange={(e) => setLocal(Number(e.target.value))}
        onPointerUp={() => onChange(local)}
        onKeyUp={() => onChange(local)}
        className="w-full accent-emerald-400"
        aria-label={label}
      />
    </div>
  );
}
