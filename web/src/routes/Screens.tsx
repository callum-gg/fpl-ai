/** The remaining screens from docs/10: squad, comparison, planner, ticker, feed,
 *  model performance, chat and settings. Grouped because each is compact. */
import { useEffect, useMemo, useState } from "react";
import {
  api,
  ALL_PLAYERS,
  CHIP_LABELS,
  CHIPS,
  matchesName,
  money,
  pts,
  useBacktests,
  useComparison,
  useFeed,
  useGlobalSettings,
  useJobs,
  useModels,
  usePatchSettings,
  usePundits,
  useRecommendations,
  useRunJob,
  useSettingsSchema,
  useSquad,
  useSources,
  useSyncSquad,
  useDraft,
  useSeedDraft,
  useEditDraft,
  useDiscardDraft,
  useCommitDraft,
  usePlayers,
  useRecommend,
  useTicker,
  useVerifyKeys,
  useWhatIf,
  type ChipUse,
  type Draft,
} from "../lib/api";
import { useSquadStore } from "../stores/squad";
import {
  Card,
  Delta,
  Empty,
  ErrorNote,
  Loading,
  PlayerName,
  PlayerRow,
  Sheet,
  Slider,
  Toggle,
} from "../components/ui";

/* ── 2. Squad view ─────────────────────────────────────────────────────────── */

/** Three ways to look at the same 15: the squad you own, a scratch copy you can
 *  rearrange, and whatever the optimiser suggests. Only "set" is real. */
type View = "set" | "working" | "recommended";

export function SquadView() {
  const { activeSquadId } = useSquadStore();
  const { data: squad } = useSquad(activeSquadId ?? undefined);
  const { data: recs } = useRecommendations(activeSquadId ?? undefined);
  const { data: draft } = useDraft(activeSquadId ?? undefined);
  const sync = useSyncSquad(activeSquadId ?? undefined);
  const seedDraft = useSeedDraft(activeSquadId ?? undefined);
  const editDraft = useEditDraft(activeSquadId ?? undefined);
  const discardDraft = useDiscardDraft(activeSquadId ?? undefined);
  const commitDraft = useCommitDraft(activeSquadId ?? undefined);
  const recommend = useRecommend(activeSquadId ?? undefined);

  const [view, setView] = useState<View>("set");
  const [pushOpen, setPushOpen] = useState(false);
  const [addOpen, setAddOpen] = useState(false);

  const rec = recs?.[0];
  const state = squad?.state;
  const editing = view === "working" && !!draft;

  // Three server shapes, one <PlayerRow> — see playerOf() in components/ui.
  const rows: any[] =
    view === "recommended" && rec
      ? rec.payload.lineup.xi
      : editing
        ? draft!.picks
        : (state?.picks ?? []);

  const shown = editing ? draft! : state;
  const busy =
    seedDraft.isPending || editDraft.isPending || commitDraft.isPending || discardDraft.isPending;
  const actionError =
    seedDraft.error ?? editDraft.error ?? commitDraft.error ?? discardDraft.error ?? sync.error;

  if (!activeSquadId) return <Empty title="No squad selected" />;

  /** Pull the recommendation's players into the working copy, keeping what already matches. */
  async function loadRecommendedIntoDraft() {
    if (!rec) return;
    if (!draft) await seedDraft.mutateAsync({ from_recommendation: rec.id });
    else {
      const want = rec.payload.lineup.xi
        .concat(rec.payload.lineup.bench_order ?? [])
        .map((p: any) => p.player_id);
      const have = draft.picks.map((p) => p.player_id);
      await editDraft.mutateAsync({
        add: want.filter((id: number) => !have.includes(id)),
        drop: have.filter((id) => !want.includes(id)),
      });
    }
    setView("working");
  }

  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      <Card
        title="Squad"
        actions={
          <div className="flex gap-2">
            <button className="btn" onClick={() => sync.mutate()} disabled={sync.isPending}>
              {sync.isPending ? "Syncing…" : "Sync from FPL"}
            </button>
            <button className="btn" onClick={() => setPushOpen(true)}>
              Push to FPL
            </button>
          </div>
        }
      >
        {actionError && (
          <div className="mb-3">
            <ErrorNote error={actionError} />
          </div>
        )}

        <div className="flex gap-2 mb-3 flex-wrap">
          {(
            [
              ["set", "Set squad"],
              ["working", draft ? "Working copy" : "Working copy (none)"],
              ["recommended", "Recommended"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              className={`chip ${key === view ? "bg-pos/15 border-pos/40 text-pos" : "text-muted"}`}
              onClick={() => setView(key as View)}
            >
              {label}
            </button>
          ))}
        </div>

        {shown && (
          <dl className="grid grid-cols-3 gap-3 mb-4 text-sm">
            <div>
              <dt className="text-muted text-xs uppercase">Bank</dt>
              <dd className="num">{money(shown.bank)}</dd>
            </div>
            <div>
              <dt className="text-muted text-xs uppercase">Value</dt>
              <dd className="num">{money(shown.squad_value)}</dd>
            </div>
            <div>
              <dt className="text-muted text-xs uppercase">Free transfers</dt>
              <dd className="num">{shown.free_transfers}</dd>
            </div>
          </dl>
        )}

        {/* A working copy is allowed to be illegal mid-edit; it just cannot be committed. */}
        {editing && draft!.errors.length > 0 && (
          <ul className="mb-3 text-sm text-warn list-disc pl-5">
            {draft!.errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        )}

        {view === "working" && !draft ? (
          <Empty
            title="No working copy"
            hint="Start one from your set squad, then swap players freely — your real squad is untouched until you commit."
          />
        ) : !rows.length ? (
          <Empty
            title={view === "recommended" ? "No recommendation yet" : "No squad set"}
            hint={
              view === "recommended"
                ? "Generate one to see a proposed 15."
                : "Sync from FPL, or build a working copy and commit it."
            }
          />
        ) : (
          <ul className="divide-y divide-line">
            {rows.map((p: any) => (
              <PlayerRow key={p.player_id} player={p}>
                {editing && (
                  <PriceCell
                    paid={p.purchase_price}
                    busy={busy}
                    onSet={(tenths) =>
                      editDraft.mutate({ prices: { [p.player_id]: tenths } })
                    }
                  />
                )}
                {editing && (
                  <button
                    className="chip text-muted hover:text-neg"
                    title="Remove from working copy"
                    disabled={busy}
                    onClick={() => editDraft.mutate({ drop: [p.player_id] })}
                  >
                    ✕
                  </button>
                )}
                {view === "recommended" && draft && (
                  <button
                    className="chip text-muted hover:text-pos"
                    title="Add to working copy"
                    disabled={busy || draft.picks.some((d) => d.player_id === p.player_id)}
                    onClick={() => editDraft.mutate({ add: [p.player_id] })}
                  >
                    +
                  </button>
                )}
              </PlayerRow>
            ))}
          </ul>
        )}

        <div className="flex gap-2 mt-4 flex-wrap">
          {!draft ? (
            <button
              className="btn"
              disabled={busy}
              // No set squad is the case that needs this most, not the one to refuse:
              // a disconnected app starts from an empty fifteen and you type in the truth.
              title={state ? "Copy your set squad" : "Build a squad from scratch"}
              onClick={async () => {
                await seedDraft.mutateAsync({});
                setView("working");
              }}
            >
              {state ? "Start working copy" : "Build squad by hand"}
            </button>
          ) : (
            <>
              <button className="btn" disabled={busy} onClick={() => setAddOpen(true)}>
                Add player
              </button>
              <button
                className="btn"
                disabled={busy || !recommend}
                onClick={() =>
                  recommend.mutate({ use_draft: true, force_refresh: true })
                }
              >
                {recommend.isPending ? "Solving…" : "Recommend from working copy"}
              </button>
              <button
                className="btn btn-primary"
                disabled={busy || !draft.ok}
                title={draft.ok ? "Make this your set squad" : draft.errors.join("; ")}
                onClick={() => commitDraft.mutate()}
              >
                Commit as set squad
              </button>
              <button className="btn" disabled={busy} onClick={() => discardDraft.mutate()}>
                Discard
              </button>
            </>
          )}
          {rec && (
            <button className="btn" disabled={busy} onClick={loadRecommendedIntoDraft}>
              Load recommended
            </button>
          )}
        </div>
      </Card>

      {/* Only alongside the copy it edits — these numbers mean nothing next to the set squad. */}
      {editing && (
        <ManualOverrides draft={draft} busy={busy} onApply={(body) => editDraft.mutate(body)} />
      )}

      <AddPlayerSheet
        open={addOpen}
        onClose={() => setAddOpen(false)}
        owned={draft?.picks.map((p) => p.player_id) ?? []}
        onPick={(id) => {
          editDraft.mutate({ add: [id] });
          setAddOpen(false);
        }}
      />

      <PushDialog open={pushOpen} onClose={() => setPushOpen(false)} recId={rec?.id} squadId={activeSquadId} />
    </div>
  );
}

/** Everything about a squad that is a number rather than a player.
 *
 * `Sync from FPL` is a guess at what FPL holds, and when it is wrong there is no argument
 * to be had with it — a stale free-transfer count silently poisons every hit the planner
 * prices, a chip it thinks you still hold gets planned into a gameweek you cannot use it,
 * and neither shows up as an error. So each of those is typed in directly, and FPL stays
 * the source of truth even when the sync is not. Nothing here touches your set squad until
 * you commit the working copy. */
function ManualOverrides({
  draft,
  busy,
  onApply,
}: {
  draft?: Draft;
  busy: boolean;
  onApply: (body: {
    gameweek?: number;
    bank?: number;
    free_transfers?: number;
    chips_used?: ChipUse[];
    chip_active?: string;
  }) => void;
}) {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<Record<string, string>>({});

  // Re-read from the draft whenever it changes underneath us (a commit, a discard, a
  // sync), so the boxes never show a value the server has already moved past.
  const serverForm = useMemo<Record<string, string>>(() => {
    if (!draft) return {};
    const used: ChipUse[] = draft.chips_used_json ? JSON.parse(draft.chips_used_json) : [];
    const chips = Object.fromEntries(
      CHIPS.map((c) => [c, String(used.find((u) => u.name === c)?.gameweek ?? "")]),
    );
    return {
      gameweek: String(draft.gameweek),
      bank: (draft.bank / 10).toFixed(1),
      free_transfers: String(draft.free_transfers),
      chip_active: draft.chip_active ?? "",
      ...chips,
    };
  }, [draft]);

  useEffect(() => setForm(serverForm), [serverForm]);

  if (!draft) return null;
  const set = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }));
  const dirty = CHIPS.concat(["gameweek", "bank", "free_transfers", "chip_active"] as any).some(
    (k) => (form[k] ?? "") !== (serverForm[k] ?? ""),
  );

  function apply() {
    onApply({
      gameweek: Number(form.gameweek) || undefined,
      bank: Math.round(Number(form.bank) * 10),
      free_transfers: Number(form.free_transfers),
      chip_active: form.chip_active,   // "" clears it; the field is always sent
      chips_used: CHIPS.filter((c) => form[c] !== "").map((c) => ({
        name: c,
        gameweek: Number(form[c]),
      })),
    });
  }

  return (
    <Card
      title="Manual overrides"
      actions={
        <button className="btn" onClick={() => setOpen((o) => !o)}>
          {open ? "Hide" : "Edit"}
        </button>
      }
    >
      {!open ? (
        <p className="text-sm text-muted">
          Bank, free transfers, gameweek and chips — set them by hand when the sync and FPL
          have drifted apart.
        </p>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Field label="Gameweek">
              <input className={INPUT} inputMode="numeric" value={form.gameweek ?? ""}
                onChange={(e) => set("gameweek", e.target.value)} />
            </Field>
            <Field label="Bank £m">
              <input className={INPUT} inputMode="decimal" value={form.bank ?? ""}
                onChange={(e) => set("bank", e.target.value)} />
            </Field>
            <Field label="Free transfers">
              <input className={INPUT} inputMode="numeric" value={form.free_transfers ?? ""}
                onChange={(e) => set("free_transfers", e.target.value)} />
            </Field>
            <Field label="Chip this GW">
              <select className={INPUT} value={form.chip_active ?? ""}
                onChange={(e) => set("chip_active", e.target.value)}>
                <option value="">None</option>
                {CHIPS.map((c) => (
                  <option key={c} value={c}>{CHIP_LABELS[c]}</option>
                ))}
              </select>
            </Field>
          </div>

          <div>
            <p className="text-muted text-xs uppercase mb-2">
              Chips already played — the gameweek, or blank if still held
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {CHIPS.map((c) => (
                <Field key={c} label={CHIP_LABELS[c]}>
                  <input className={INPUT} inputMode="numeric" placeholder="unused"
                    value={form[c] ?? ""} onChange={(e) => set(c, e.target.value)} />
                </Field>
              ))}
            </div>
          </div>

          <button className="btn btn-primary" disabled={busy || !dirty} onClick={apply}>
            {busy ? "Saving…" : "Apply to working copy"}
          </button>
        </div>
      )}
    </Card>
  );
}

/** What you paid for one player, editable in place.
 *
 * Adding a player prices him at today's value, which is right for a new signing and wrong
 * for every player you already held — you keep only half of any rise, so a squad rebuilt at
 * today's prices reports a selling value you cannot actually realise. Committed on blur,
 * because a PATCH per keystroke would fight the cursor. */
function PriceCell({
  paid,
  busy,
  onSet,
}: {
  paid: number | null;
  busy: boolean;
  onSet: (tenths: number) => void;
}) {
  const asText = paid == null ? "" : (paid / 10).toFixed(1);
  const [text, setText] = useState(asText);
  useEffect(() => setText(asText), [asText]);

  return (
    <input
      className="bg-raised border border-line rounded px-2 py-1 text-xs w-16 num text-right"
      inputMode="decimal"
      title="Purchase price — what you actually paid"
      disabled={busy}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        const tenths = Math.round(Number(text) * 10);
        if (text !== asText && tenths > 0) onSet(tenths);
        else setText(asText);
      }}
    />
  );
}

const INPUT = "bg-raised border border-line rounded-lg px-3 py-2 text-sm w-full";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-muted text-xs uppercase block mb-1">{label}</span>
      {children}
    </label>
  );
}

/** Searchable, whole-league player list for building a working copy by hand.
 *
 * It used to open on MID and ask for 40 rows, which is a top-40 leaderboard wearing a
 * player picker's clothes: the enabler you actually wanted was never on it, and there was
 * no position for "all". A hand-built squad needs every player reachable, so the list is
 * the league and the search box is how you get through it — same client-side filter the
 * explorer screen uses, over the same 600-row fetch. */
function AddPlayerSheet({
  open,
  onClose,
  owned,
  onPick,
}: {
  open: boolean;
  onClose: () => void;
  owned: number[];
  onPick: (playerId: number) => void;
}) {
  const [position, setPosition] = useState<string>("");
  const [term, setTerm] = useState("");
  const { data: players, isLoading } = usePlayers({
    position: position || undefined,
    limit: ALL_PLAYERS,
  });

  const rows = useMemo(
    () => (players ?? []).filter((p) => matchesName(p, term)),
    [players, term],
  );

  return (
    <Sheet open={open} onClose={onClose} title="Add player">
      <input
        className="bg-raised border border-line rounded-lg px-3 py-2 text-sm w-full mb-3"
        placeholder="Search name"
        value={term}
        onChange={(e) => setTerm(e.target.value)}
        autoFocus
      />
      <div className="flex gap-2 mb-3">
        {["", "GK", "DEF", "MID", "FWD"].map((p) => (
          <button
            key={p || "all"}
            className={`chip ${p === position ? "bg-pos/15 border-pos/40 text-pos" : "text-muted"}`}
            onClick={() => setPosition(p)}
          >
            {p || "All"}
          </button>
        ))}
      </div>
      {isLoading ? (
        <Loading />
      ) : rows.length === 0 ? (
        <Empty title={term ? `No player matches "${term}"` : "No players"} />
      ) : (
        <ul className="divide-y divide-line max-h-96 overflow-y-auto">
          {rows.map((p) => (
            <PlayerRow key={p.id} player={p}>
              <button
                className="chip"
                disabled={owned.includes(p.id)}
                onClick={() => onPick(p.id)}
              >
                {owned.includes(p.id) ? "in" : "+"}
              </button>
            </PlayerRow>
          ))}
        </ul>
      )}
    </Sheet>
  );
}

/** Push needs a preview, a typed confirmation, and an explicit refusal path. */
function PushDialog({
  open,
  onClose,
  recId,
  squadId,
}: {
  open: boolean;
  onClose: () => void;
  recId?: number;
  squadId: number;
}) {
  const [preview, setPreview] = useState<any>(null);
  const [confirmation, setConfirmation] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  async function loadPreview() {
    setError(null);
    try {
      setPreview(await api.post(`/api/squads/${squadId}/push/preview?recommendation_id=${recId}`));
    } catch (e) {
      setError(e);
    }
  }

  async function execute(dryRun: boolean) {
    setError(null);
    try {
      const r: any = await api.post(`/api/squads/${squadId}/push/execute`, {
        recommendation_id: recId,
        confirmation_text: confirmation,
        preview_generated_at: preview?.generated_at,
        dry_run: dryRun,
      });
      setResult(dryRun ? "Dry run OK — nothing was sent." : "Pushed to FPL.");
      if (!dryRun) setTimeout(onClose, 1500);
      return r;
    } catch (e) {
      setError(e);
    }
  }

  return (
    <Sheet open={open} onClose={onClose} title="Push to FPL">
      {!recId ? (
        <Empty title="Generate a recommendation first" />
      ) : !preview ? (
        <div className="space-y-3">
          <p className="text-sm text-muted">
            This writes to your real FPL team through an unofficial endpoint. It always
            previews first, and a scheduled job can never do it.
          </p>
          <button className="btn btn-primary" onClick={loadPreview}>
            Generate preview
          </button>
        </div>
      ) : (
        <div className="space-y-3 text-sm">
          {preview.warnings?.map((w: string, i: number) => (
            <p key={i} className="text-warn">
              {w}
            </p>
          ))}
          <div>
            <p className="text-muted text-xs uppercase mb-1">In</p>
            <ul>
              {preview.transfers_diff?.in?.map((p: any) => (
                <li key={p.player_id}>
                  <PlayerName player={p} />
                </li>
              ))}
            </ul>
            <p className="text-muted text-xs uppercase mt-2 mb-1">Out</p>
            <ul>
              {preview.transfers_diff?.out?.map((p: any) => (
                <li key={p.player_id}>
                  <PlayerName player={p} />
                </li>
              ))}
            </ul>
          </div>
          <p>
            Cost: <span className="num">{preview.cost}</span> pts
          </p>
          <label className="block">
            <span className="text-muted text-xs">
              Type <code className="text-slate-200">{preview.confirmation_required}</code> to confirm
            </span>
            <input
              className="w-full bg-raised border border-line rounded-lg px-3 py-2 mt-1"
              value={confirmation}
              onChange={(e) => setConfirmation(e.target.value)}
            />
          </label>
          <div className="flex gap-2">
            <button className="btn" onClick={() => execute(true)}>
              Dry run
            </button>
            <button
              className="btn btn-primary"
              disabled={confirmation !== preview.confirmation_required}
              onClick={() => execute(false)}
            >
              Push for real
            </button>
          </div>
          {result && <p className="text-pos">{result}</p>}
          <ErrorNote error={error} />
        </div>
      )}
    </Sheet>
  );
}

/* ── 3. Comparison ─────────────────────────────────────────────────────────── */

export function Compare() {
  const { comparisonSet } = useSquadStore();
  const { data, isLoading } = useComparison(comparisonSet);

  if (comparisonSet.length < 2)
    return <Empty title="Pick at least two squads" hint="Use the 'cmp' checkboxes in the squad picker." />;
  if (isLoading) return <Loading />;

  const metrics: [string, (c: any) => string][] = [
    ["Projected GW", (c) => pts(c.exp_points_gw)],
    ["Horizon", (c) => pts(c.exp_points_horizon)],
    ["Spread (SD)", (c) => `±${pts(c.sd_points_gw)}`],
    ["Risk", (c) => String(c.risk ?? "—")],
    ["Rank mode", (c) => c.rank_mode ?? "—"],
    ["Hits", (c) => String(c.hits ?? 0)],
    ["Chip", (c) => c.chip ?? "—"],
    ["Transfers", (c) => String(c.transfers?.length ?? 0)],
  ];

  return (
    <div className="space-y-4">
      <Card title={`Comparison · GW${data?.gameweek}`}>
        <div className="scroll-x">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className="text-left sticky left-0 bg-surface pr-3">Metric</th>
                {data.columns.map((c: any) => (
                  <th key={c.squad_id} className="text-right px-3 whitespace-nowrap">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full" style={{ background: c.colour }} />
                      {c.name}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.map(([label, fn]) => (
                <tr key={label} className="border-t border-line">
                  <td className="py-2 sticky left-0 bg-surface text-muted pr-3">{label}</td>
                  {data.columns.map((c: any) => (
                    <td key={c.squad_id} className="text-right num py-2 px-3">
                      {fn(c)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted mt-3">
          {data.shared_players?.length ?? 0} players shared across all squads.
        </p>
      </Card>

      {data.columns.map((c: any) => (
        <Card key={c.squad_id} title={c.name}>
          <p className="text-sm">{c.headline ?? "No recommendation yet."}</p>
        </Card>
      ))}
    </div>
  );
}

/* ── 6. Transfer planner ───────────────────────────────────────────────────── */

export function Planner() {
  const { activeSquadId } = useSquadStore();
  const { data: recs } = useRecommendations(activeSquadId ?? undefined);
  const whatIf = useWhatIf(activeSquadId ?? undefined);
  const [forceOut, setForceOut] = useState<number[]>([]);
  const rec = recs?.[0];

  if (!rec) return <Empty title="No plan yet" hint="Generate a recommendation on the dashboard." />;

  const gws = [
    { ...rec.payload, exp_points: rec.payload.totals.exp_points_gw },
    ...rec.payload.future_gameweeks,
  ];

  return (
    <div className="space-y-4">
      <Card title="Multi-gameweek plan">
        <div className="scroll-x">
          <table className="w-full text-sm">
            <thead className="text-muted text-xs uppercase">
              <tr>
                <th className="text-left py-2">GW</th>
                <th className="text-left py-2">In</th>
                <th className="text-left py-2">Out</th>
                <th className="text-right py-2">Hits</th>
                <th className="text-right py-2">Chip</th>
                <th className="text-right py-2">EP</th>
              </tr>
            </thead>
            <tbody>
              {gws.map((g: any) => (
                <tr key={g.gameweek} className="border-t border-line">
                  <td className="py-2 num">{g.gameweek}</td>
                  <td className="py-2 text-pos">
                    {(g.transfers_in ?? g.transfers?.map((t: any) => t.in) ?? [])
                      .map((p: any) => p.name)
                      .join(", ") || "—"}
                  </td>
                  <td className="py-2 text-neg">
                    {(g.transfers_out ?? g.transfers?.map((t: any) => t.out) ?? [])
                      .map((p: any) => p.name)
                      .join(", ") || "—"}
                  </td>
                  <td className="py-2 text-right num">{g.hits ?? 0}</td>
                  <td className="py-2 text-right">{g.chip ?? "—"}</td>
                  <td className="py-2 text-right num">{pts(g.exp_points)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="What if?">
        <p className="text-sm text-muted mb-3">
          Force a player out and the optimiser re-solves the rest around your constraint,
          reporting the honest cost of your idea in points.
        </p>
        <div className="flex flex-wrap gap-1.5 mb-3">
          {rec.payload.squad.map((p) => (
            <button
              key={p.player_id}
              className={`chip ${forceOut.includes(p.player_id) ? "bg-neg/15 border-neg/40 text-neg" : "text-muted"}`}
              onClick={() =>
                setForceOut((f) =>
                  f.includes(p.player_id) ? f.filter((x) => x !== p.player_id) : [...f, p.player_id],
                )
              }
            >
              <PlayerName player={p} />
            </button>
          ))}
        </div>
        <button
          className="btn btn-primary"
          disabled={!forceOut.length || whatIf.isPending}
          onClick={() => whatIf.mutate({ force_out: forceOut })}
        >
          {whatIf.isPending ? "Re-solving…" : "Re-solve without them"}
        </button>
        <ErrorNote error={whatIf.error} />
        {whatIf.data && (
          <div className="mt-3 text-sm">
            <p>{whatIf.data.payload.headline}</p>
            <p className="text-muted mt-1">
              Horizon {pts(whatIf.data.payload.totals.exp_points_horizon)} vs{" "}
              {pts(rec.payload.totals.exp_points_horizon)} —{" "}
              <Delta
                value={
                  whatIf.data.payload.totals.exp_points_horizon -
                  rec.payload.totals.exp_points_horizon
                }
              />
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}

/* ── 7. Fixture ticker ─────────────────────────────────────────────────────── */

export function Ticker() {
  const { data, isLoading } = useTicker("1-8");
  if (isLoading) return <Loading />;
  if (!data?.teams?.length) return <Empty title="No fixtures loaded" />;

  const tone = (d: number) =>
    d <= 0.9 ? "bg-pos/25" : d <= 1.3 ? "bg-pos/10" : d <= 1.7 ? "bg-slate-600/30" : "bg-neg/20";

  return (
    <Card title="Fixture ticker — model difficulty, not FDR">
      <div className="scroll-x">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="sticky left-0 bg-surface text-left p-2">Team</th>
              {data.gameweeks.map((g: number) => (
                <th key={g} className="p-2 num">
                  {g}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.teams.map((t: any) => (
              <tr key={t.id}>
                <td className="sticky left-0 bg-surface p-2 whitespace-nowrap">{t.short_name}</td>
                {data.gameweeks.map((g: number) => {
                  const cell = t.fixtures[String(g)];
                  return (
                    <td key={g} className="p-1">
                      {cell.bgw ? (
                        <div className="h-7 w-12 rounded bg-neg/30 grid place-items-center text-[10px]">
                          BGW
                        </div>
                      ) : (
                        <div className="flex flex-col gap-0.5">
                          {cell.matches.map((m: any, i: number) => (
                            <div
                              key={i}
                              className={`h-7 w-12 rounded grid place-items-center ${tone(m.model_difficulty)}`}
                              title={`xG for ${m.expected_goals}, against ${m.model_difficulty}`}
                            >
                              {m.is_home ? "H" : "A"}
                              {cell.dgw && <sup className="text-pos">2</sup>}
                            </div>
                          ))}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

/* ── 8. Feed ───────────────────────────────────────────────────────────────── */

export function Feed() {
  const [types, setTypes] = useState("article,video,social,claim");
  const { data, isLoading } = useFeed(types);
  const { data: sources } = useSources();
  const stale = (sources ?? []).filter((s) => s.enabled && s.available && !s.last_run);

  return (
    <div className="space-y-4">
      <Card title="Filter">
        <div className="flex flex-wrap gap-2">
          {["article", "video", "social", "claim"].map((t) => (
            <button
              key={t}
              className={`chip capitalize ${types.includes(t) ? "bg-pos/15 border-pos/40 text-pos" : "text-muted"}`}
              onClick={() =>
                setTypes((cur) => {
                  const set = new Set(cur.split(",").filter(Boolean));
                  set.has(t) ? set.delete(t) : set.add(t);
                  return [...set].join(",");
                })
              }
            >
              {t}
            </button>
          ))}
        </div>
      </Card>

      {stale.length > 0 && (
        <Card title="Source health">
          <p className="text-sm text-warn">
            {stale.length} enabled source{stale.length > 1 ? "s have" : " has"} never run:{" "}
            {stale.map((s) => s.id).join(", ")}.
          </p>
        </Card>
      )}

      {isLoading ? (
        <Loading />
      ) : !data?.length ? (
        <Empty title="Nothing ingested yet" hint="Run an ingest job from Settings → Jobs." />
      ) : (
        <ul className="space-y-2">
          {data.map((item) => (
            <li key={`${item.type}-${item.id}`} className="card p-3 feed">
              <div className="flex items-center gap-2 text-xs text-muted mb-1 flex-wrap">
                <span className="chip">{item.type}</span>
                <span>{item.outlet ?? item.channel_title ?? item.platform ?? item.source_id}</span>
                {item.published_at && <span>{item.published_at.slice(0, 16).replace("T", " ")}</span>}
                {item.seen_on_sites && item.seen_on_sites > 1 && (
                  <span className="chip text-warn">seen on {item.seen_on_sites} sites</span>
                )}
                {item.retrieval_method && <span className="chip">via {item.retrieval_method}</span>}
              </div>
              <p className="text-sm">
                {item.title ?? item.text_span ?? item.body_text?.slice(0, 200) ?? "—"}
              </p>
              {item.player_name && (
                <p className="text-xs text-muted mt-1">
                  {item.player_name} · {item.claim_type} · {item.stance}
                </p>
              )}
              {item.url && (
                <a href={item.url} target="_blank" rel="noreferrer" className="text-xs text-pos hover:underline">
                  open
                </a>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ── 9. Model performance ──────────────────────────────────────────────────── */

export function Performance() {
  const { data: models } = useModels();
  const { data: pundits } = usePundits();
  const { data: backtests } = useBacktests();
  const active = (models ?? []).filter((m: any) => m.is_active);

  return (
    <div className="space-y-4">
      <Card title="Active models">
        {!active.length ? (
          <Empty title="Nothing trained yet" hint="Run `fplai train` or Settings → Jobs → train_models." />
        ) : (
          <ul className="space-y-2 text-sm">
            {active.map((m: any) => (
              <li key={m.id} className="flex justify-between items-start gap-3">
                <div>
                  <span className="font-medium">{m.model_name}</span>{" "}
                  <span className="text-muted text-xs">v{m.version}</span>
                  {m.metrics?.regime_warning && (
                    <p className="text-warn text-xs mt-0.5">{m.metrics.regime_warning}</p>
                  )}
                </div>
                <span className="num text-muted text-xs whitespace-nowrap">
                  {["log_loss", "mae", "spearman", "calibration_ece"]
                    .filter((k) => m.metrics?.[k] != null)
                    .map((k) => `${k} ${Number(m.metrics[k]).toFixed(3)}`)
                    .join(" · ")}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Backtests">
        {!backtests?.length ? (
          <Empty title="No backtests yet" />
        ) : (
          <ul className="space-y-2 text-sm">
            {backtests.map((b: any) => (
              <li key={b.id} className="flex justify-between">
                <span>{b.seasons}</span>
                <span className="num">
                  {b.total_points} pts · avg {b.avg_gw_points} · {b.hits_taken} hits
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Pundit accuracy">
        <p className="text-xs text-muted mb-2">{pundits?.note}</p>
        {!pundits?.scoreboard?.length ? (
          <Empty title="No resolved calls yet" hint="Needs claim extraction plus a finished gameweek." />
        ) : (
          <table className="w-full text-sm">
            <thead className="text-muted text-xs uppercase">
              <tr>
                <th className="text-left py-1">Source</th>
                <th className="text-right py-1">Calls</th>
                <th className="text-right py-1">Avg pts</th>
                <th className="text-right py-1">vs baseline</th>
              </tr>
            </thead>
            <tbody>
              {pundits.scoreboard.map((p: any) => (
                <tr key={p.name} className="border-t border-line">
                  <td className="py-1">{p.name}</td>
                  <td className="text-right num py-1">{p.calls}</td>
                  <td className="text-right num py-1">{p.avg_points}</td>
                  <td className="text-right num py-1">
                    <Delta value={p.avg_score} dp={2} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

/* ── 10. Chat ──────────────────────────────────────────────────────────────── */

export function Chat() {
  const { activeSquadId } = useSquadStore();
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [steps, setSteps] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function send() {
    if (!input.trim()) return;
    const next = [...messages, { role: "user", content: input }];
    setMessages(next);
    setInput("");
    setBusy(true);
    setSteps([]);

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ squad_id: activeSquadId, messages: next }),
    });
    const reader = res.body?.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (reader) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      for (const line of buffer.split("\n\n")) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;
        try {
          const step = JSON.parse(payload);
          if (step.type === "tool") setSteps((s) => [...s, step.summary]);
          if (step.type === "message")
            setMessages((m) => [...m, { role: "assistant", content: step.content }]);
          if (step.type === "error")
            setMessages((m) => [...m, { role: "assistant", content: `⚠ ${step.message}` }]);
        } catch {
          /* partial frame; wait for more */
        }
      }
      buffer = buffer.slice(buffer.lastIndexOf("\n\n") + 2);
    }
    setBusy(false);
  }

  return (
    <div className="flex flex-col h-[calc(100dvh-9rem-env(safe-area-inset-top,0px)-env(safe-area-inset-bottom,0px))] max-w-3xl mx-auto">
      <div className="flex-1 min-h-0 overflow-y-auto space-y-3 pb-3">
        {!messages.length && (
          <Card title="Try asking">
            <ul className="text-sm space-y-1 text-muted">
              <li>“Why not Haaland this week?”</li>
              <li>“What if I go without a premium striker?”</li>
              <li>“Compare my two squads for GW14.”</li>
            </ul>
          </Card>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`card p-3 text-sm whitespace-pre-wrap ${
              m.role === "user" ? "bg-raised ml-8" : "mr-8"
            }`}
          >
            {m.content}
          </div>
        ))}
        {steps.length > 0 && busy && (
          <details className="text-xs text-muted" open>
            <summary>{steps.length} tool steps</summary>
            <ul>{steps.map((s, i) => <li key={i}>{s}</li>)}</ul>
          </details>
        )}
      </div>
      <form
        className="flex gap-2 pt-2 border-t border-line"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          className="flex-1 bg-raised border border-line rounded-lg px-3 py-2 text-sm"
          placeholder="Ask about your squad…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        <button className="btn btn-primary" disabled={busy}>
          {busy ? "…" : "Send"}
        </button>
      </form>
      <p className="text-[11px] text-muted mt-1">
        Read-only: chat can propose changes but never applies them.
      </p>
    </div>
  );
}

/* ── 11. Settings ──────────────────────────────────────────────────────────── */

export function Settings() {
  const [tab, setTab] = useState<"global" | "squad" | "sources" | "jobs">("global");
  const { activeSquadId } = useSquadStore();
  const { data: schema } = useSettingsSchema();
  const { data: global } = useGlobalSettings();
  const { data: squad } = useSquad(activeSquadId ?? undefined);
  const patchGlobal = usePatchSettings("global");
  const patchSquad = usePatchSettings(activeSquadId ?? 0);
  const { data: sources } = useSources();
  const { data: jobs } = useJobs();
  const runJob = useRunJob();
  const verifyKeys = useVerifyKeys();

  const render = (fields: any[], values: Record<string, any>, patch: any) =>
    fields.map((f) => {
      const value = values?.[f.key] ?? f.default;
      if (f.widget === "slider")
        return (
          <Slider
            key={f.key}
            label={f.key}
            min={f.min}
            max={f.max}
            step={f.step}
            value={Number(value)}
            onChange={(v) => patch.mutate({ [f.key]: v })}
          />
        );
      if (f.type === "boolean")
        return (
          <Toggle
            key={f.key}
            label={f.key}
            checked={Boolean(value)}
            onChange={(v) => patch.mutate({ [f.key]: v })}
          />
        );
      if (f.widget === "select")
        return (
          <label key={f.key} className="flex justify-between items-center py-1.5 text-sm">
            <span>{f.key}</span>
            <select
              className="bg-raised border border-line rounded px-2 py-1"
              value={String(value)}
              onChange={(e) => patch.mutate({ [f.key]: e.target.value })}
            >
              {f.options.map((o: string) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </label>
        );
      if (f.type === "number")
        return (
          <label key={f.key} className="flex justify-between items-center py-1.5 text-sm">
            <span>{f.key}</span>
            <input
              type="number"
              className="bg-raised border border-line rounded px-2 py-1 w-24 num"
              defaultValue={Number(value)}
              onBlur={(e) => patch.mutate({ [f.key]: Number(e.target.value) })}
            />
          </label>
        );
      return (
        <div key={f.key} className="py-1.5 text-sm flex justify-between gap-3">
          <span className="text-muted">{f.key}</span>
          <span className="text-xs text-muted truncate max-w-[60%]">
            {Array.isArray(value) ? `${value.length} items` : typeof value === "object" ? "object" : String(value)}
          </span>
        </div>
      );
    });

  return (
    <div className="space-y-4 max-w-3xl mx-auto">
      <div className="flex gap-2 flex-wrap">
        {(["global", "squad", "sources", "jobs"] as const).map((t) => (
          <button
            key={t}
            className={`chip capitalize ${t === tab ? "bg-pos/15 border-pos/40 text-pos" : "text-muted"}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "global" && schema && (
        <Card title="Global settings">{render(schema.global, global?.settings ?? {}, patchGlobal)}</Card>
      )}

      {tab === "global" && global?.env && (
        <Card title="From .env (read-only)">
          <ul className="text-xs space-y-0.5">
            {global.env
              .filter((e: any) => e.secret)
              .map((e: any) => (
                <li key={e.key} className="flex justify-between">
                  <span className="text-muted">{e.key.toUpperCase()}</span>
                  <span className={e.value ? "text-pos" : "text-muted"}>
                    {e.value ? "set" : "not set"}
                  </span>
                </li>
              ))}
          </ul>
          <div className="mt-3 pt-3 border-t border-line">
            <button
              className="btn text-xs"
              onClick={() => verifyKeys.mutate()}
              disabled={verifyKeys.isPending}
            >
              {verifyKeys.isPending ? "Verifying…" : "Verify keys"}
            </button>
            <p className="text-[11px] text-muted mt-1">
              Makes a real, cheap call to each configured service — presence alone
              doesn't prove a key still works.
            </p>
            {verifyKeys.data && (
              <ul className="text-xs space-y-0.5 mt-2">
                {verifyKeys.data.results.length === 0 && (
                  <li className="text-muted">no credentialed services configured</li>
                )}
                {verifyKeys.data.results.map((r) => (
                  <li key={r.key} className="flex justify-between gap-2">
                    <span>{r.service}</span>
                    <span className={r.ok ? "text-pos" : "text-neg"}>{r.detail}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Card>
      )}

      {tab === "squad" && schema && activeSquadId && (
        <Card title={`Squad settings · ${squad?.name ?? ""}`}>
          {render(schema.squad, (squad?.settings as any) ?? {}, patchSquad)}
        </Card>
      )}

      {tab === "sources" && (
        <Card title="Sources">
          <ul className="text-sm divide-y divide-line">
            {(sources ?? []).map((s) => (
              <li key={s.id} className="py-2 flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${
                    !s.enabled ? "bg-slate-600" : s.available ? "bg-pos" : "bg-warn"
                  }`}
                />
                <span className="flex-1">
                  {s.id}
                  <span className="text-muted text-xs ml-2">{s.category}</span>
                </span>
                <span className="text-xs text-muted text-right">
                  {s.unavailable_reason ?? s.last_run?.started_at?.slice(0, 16) ?? "never run"}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {tab === "jobs" && (
        <Card title="Jobs">
          <ul className="text-sm divide-y divide-line">
            {(jobs?.registered ?? []).map((name: string) => (
              <li key={name} className="py-1.5 flex items-center gap-2">
                <span className="flex-1">{name}</span>
                <button className="btn text-xs" onClick={() => runJob.mutate(name)} disabled={runJob.isPending}>
                  Run
                </button>
              </li>
            ))}
          </ul>
          {jobs?.recent?.length > 0 && (
            <details className="mt-3 text-xs">
              <summary className="text-muted cursor-pointer">Recent runs</summary>
              <ul className="mt-1 space-y-0.5">
                {jobs.recent.slice(0, 20).map((r: any) => (
                  <li key={r.id} className="flex justify-between">
                    <span>{r.job_name}</span>
                    <span className={r.status === "failed" ? "text-neg" : "text-muted"}>
                      {r.status} · {r.started_at?.slice(11, 16)}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </Card>
      )}
    </div>
  );
}

export function More() {
  return (
    <Card title="More">
      <ul className="space-y-1">
        {[
          ["/planner", "Transfer planner"],
          ["/compare", "Compare squads"],
          ["/ticker", "Fixture ticker"],
          ["/feed", "Feed"],
          ["/performance", "Model performance"],
          ["/settings", "Settings"],
        ].map(([to, label]) => (
          <li key={to}>
            <a href={to} className="block px-3 py-2 rounded-lg hover:bg-raised">
              {label}
            </a>
          </li>
        ))}
      </ul>
    </Card>
  );
}
