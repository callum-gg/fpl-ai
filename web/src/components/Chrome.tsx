/** Global chrome: squad picker, deadline countdown, command palette, navigation. docs/10. */
import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  countdown,
  deadlineTone,
  pts,
  useCreateSquad,
  useDeadline,
  useDeleteSquad,
  usePlayers,
  useSquads,
  type Squad,
} from "../lib/api";
import { useSquadStore } from "../stores/squad";
import { Sheet } from "./ui";

const NAV = [
  { to: "/", label: "Dashboard", icon: "◎" },
  { to: "/squad", label: "Squad", icon: "⬢" },
  { to: "/players", label: "Players", icon: "☰" },
  { to: "/chat", label: "Chat", icon: "✦" },
  { to: "/more", label: "More", icon: "⋯" },
];

const MORE = [
  { to: "/planner", label: "Transfer planner", icon: "⇄" },
  { to: "/compare", label: "Compare squads", icon: "⚖" },
  { to: "/ticker", label: "Fixture ticker", icon: "▦" },
  { to: "/feed", label: "Feed", icon: "▤" },
  { to: "/performance", label: "Model performance", icon: "◐" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];

export function DeadlineCountdown() {
  const { data } = useDeadline();
  return (
    <div className="text-right leading-tight">
      <div className="text-[10px] uppercase tracking-wide text-muted">GW{data?.gameweek ?? "—"}</div>
      <div className={`num text-sm font-semibold ${deadlineTone(data?.seconds_remaining)}`}>
        {countdown(data?.seconds_remaining)}
      </div>
    </div>
  );
}

export function SquadPicker() {
  const { data: squads } = useSquads();
  const { activeSquadId, setActive, comparisonSet, toggleComparison } = useSquadStore();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [entryId, setEntryId] = useState("");
  const [confirmId, setConfirmId] = useState<number | null>(null);
  const create = useCreateSquad();
  const del = useDeleteSquad();

  const active = squads?.find((s) => s.id === activeSquadId) ?? squads?.[0];
  useEffect(() => {
    if (!activeSquadId && squads?.length) setActive(squads[0].id);
  }, [squads, activeSquadId, setActive]);

  function handleDelete(id: number) {
    del.mutate(id, {
      onSuccess: () => {
        if (activeSquadId === id) {
          const next = squads?.find((s) => s.id !== id);
          if (next) setActive(next.id);
        }
        setConfirmId(null);
      },
    });
  }

  return (
    <>
      <button
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface border border-line text-sm"
        onClick={() => setOpen(true)}
      >
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ background: active?.colour ?? "#4ADE80" }}
        />
        <span className="font-medium truncate max-w-[9rem]">{active?.name ?? "No squad"}</span>
        {active?.projected_points != null && (
          <span className="num text-muted">{pts(active.projected_points)}</span>
        )}
      </button>

      <Sheet open={open} onClose={() => setOpen(false)} title="Squads">
        <ul className="space-y-2">
          {squads?.map((s: Squad) =>
            confirmId === s.id ? (
              <li
                key={s.id}
                className="flex items-center justify-between gap-2 p-2 rounded-lg bg-neg/10 border border-neg/30 text-sm"
              >
                <span className="text-neg truncate">Delete "{s.name}"?</span>
                <div className="flex gap-1 shrink-0">
                  <button type="button" className="btn text-xs" onClick={() => setConfirmId(null)}>
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="btn text-xs bg-neg/15 border-neg/40 text-neg"
                    disabled={del.isPending}
                    onClick={() => handleDelete(s.id)}
                  >
                    {del.isPending ? "…" : "Delete"}
                  </button>
                </div>
              </li>
            ) : (
              <li key={s.id} className="flex items-center gap-2">
                <button
                  className={`flex-1 flex items-center gap-2 p-2 rounded-lg text-left ${
                    s.id === active?.id ? "bg-raised" : "hover:bg-raised"
                  }`}
                  onClick={() => {
                    setActive(s.id);
                    setOpen(false);
                  }}
                >
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: s.colour ?? "#4ADE80" }} />
                  <span className="flex-1">{s.name}</span>
                  <span className="num text-muted text-sm">{pts(s.projected_points)}</span>
                </button>
                <label className="flex items-center gap-1 text-xs text-muted">
                  <input
                    type="checkbox"
                    checked={comparisonSet.includes(s.id)}
                    onChange={() => toggleComparison(s.id)}
                    aria-label={`Compare ${s.name}`}
                  />
                  cmp
                </label>
                <button
                  type="button"
                  className="text-muted hover:text-neg px-1"
                  aria-label={`Delete ${s.name}`}
                  onClick={() => setConfirmId(s.id)}
                >
                  ✕
                </button>
              </li>
            ),
          )}
        </ul>

        <form
          className="mt-4 pt-4 border-t border-line space-y-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (!name.trim()) return;
            create.mutate(
              { name, fpl_entry_id: entryId ? Number(entryId) : undefined },
              { onSuccess: () => { setName(""); setEntryId(""); } },
            );
          }}
        >
          <p className="text-xs text-muted">
            New squad. Link an FPL entry id to sync your real team, or leave it blank for a
            shadow squad.
          </p>
          <div className="flex flex-col gap-2">
            <input
              className="w-full bg-raised border border-line rounded-lg px-3 py-2 text-sm"
              placeholder="Squad name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <input
              className="w-full bg-raised border border-line rounded-lg px-3 py-2 text-sm num"
              placeholder="entry id"
              value={entryId}
              onChange={(e) => setEntryId(e.target.value)}
            />
            <button className="btn btn-primary w-full" disabled={create.isPending}>
              Add
            </button>
          </div>
        </form>
      </Sheet>
    </>
  );
}

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [term, setTerm] = useState("");
  const navigate = useNavigate();
  const { data: players } = usePlayers({ limit: 400 });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const results = useMemo(() => {
    const t = term.toLowerCase().trim();
    const screens = [...NAV, ...MORE]
      .filter((s) => !t || s.label.toLowerCase().includes(t))
      .map((s) => ({ kind: "screen" as const, label: s.label, to: s.to }));
    const people = !t
      ? []
      : (players ?? [])
          .filter((p) => p.web_name.toLowerCase().includes(t))
          .slice(0, 8)
          .map((p) => ({
            kind: "player" as const,
            label: `${p.web_name} · ${p.team_short}`,
            to: `/players/${p.id}`,
          }));
    return [...people, ...screens].slice(0, 12);
  }, [term, players]);

  return (
    <Sheet open={open} onClose={() => setOpen(false)} title="Jump to">
      <input
        autoFocus
        className="w-full bg-raised border border-line rounded-lg px-3 py-2 mb-3"
        placeholder="Player, screen, or action…"
        value={term}
        onChange={(e) => setTerm(e.target.value)}
      />
      <ul className="space-y-1">
        {results.map((r, i) => (
          <li key={i}>
            <button
              className="w-full text-left px-3 py-2 rounded-lg hover:bg-raised flex justify-between"
              onClick={() => {
                navigate(r.to);
                setOpen(false);
                setTerm("");
              }}
            >
              <span>{r.label}</span>
              <span className="text-muted text-xs uppercase">{r.kind}</span>
            </button>
          </li>
        ))}
      </ul>
    </Sheet>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();
  return (
    <div className="min-h-full flex flex-col lg:flex-row">
      {/* Desktop sidebar: pinned to the viewport so it never scrolls with page content. */}
      <aside className="hidden lg:flex lg:w-56 lg:h-screen lg:sticky lg:top-0 flex-col border-r border-line p-4 gap-4 shrink-0 overflow-y-auto">
        <div className="flex items-center justify-between">
          <span className="font-semibold tracking-tight">FPL&nbsp;AI</span>
          <DeadlineCountdown />
        </div>
        <SquadPicker />
        <nav className="flex flex-col gap-0.5 mt-2">
          {[...NAV.slice(0, 4), ...MORE].map((n) => (
            <Link
              key={n.to}
              to={n.to}
              className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm ${
                pathname === n.to ? "bg-raised text-slate-100" : "text-muted hover:bg-raised"
              }`}
            >
              <span className="text-[1rem] leading-none w-4 shrink-0 text-center" aria-hidden="true">
                {n.icon}
              </span>
              {n.label}
            </Link>
          ))}
        </nav>
        <p className="mt-auto text-[11px] text-muted leading-snug">
          Press <kbd className="px-1 bg-raised rounded">⌘K</kbd> to jump anywhere.
        </p>
      </aside>

      {/* Mobile top bar. viewport-fit=cover means content can render under the
          notch/Dynamic Island — pt adds the real inset on top of the usual py-2. */}
      <header className="lg:hidden sticky top-0 z-30 bg-base/95 backdrop-blur border-b border-line px-3 pb-2 pt-[calc(0.5rem+env(safe-area-inset-top,0px))] flex items-center justify-between gap-2">
        <SquadPicker />
        <DeadlineCountdown />
      </header>

      <main className="flex-1 min-w-0 p-3 sm:p-4 lg:p-6 pb-[calc(5rem+env(safe-area-inset-bottom,0px))] lg:pb-6">{children}</main>

      {/* Mobile bottom tabs — every primary action in the bottom third, one-thumbed.
          pb adds the home-indicator inset so tabs don't render under it. */}
      <nav className="lg:hidden fixed bottom-0 inset-x-0 z-30 bg-base/95 backdrop-blur border-t border-line grid grid-cols-5 pb-[env(safe-area-inset-bottom,0px)]">
        {NAV.map((n) => (
          <Link
            key={n.to}
            to={n.to}
            className={`flex flex-col items-center py-2 text-[10px] ${
              pathname === n.to ? "text-pos" : "text-muted"
            }`}
          >
            <span className="text-[1rem] leading-none">{n.icon}</span>
            {n.label}
          </Link>
        ))}
      </nav>

      <CommandPalette />
    </div>
  );
}

export { MORE };
