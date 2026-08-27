/**
 * Thin fetch wrapper + typed hooks. Response shapes come from `openapi-typescript`
 * (`npm run gen:api` writes src/lib/api.d.ts); nothing here is hand-typed beyond the
 * view models the screens actually consume.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { UseQueryOptions } from "@tanstack/react-query";

const BASE = import.meta.env.VITE_API_BASE ?? "";
const TOKEN_KEY = "fplai.token";

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = localStorage.getItem(TOKEN_KEY);
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { "X-App-Token": token } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    // FastAPI nests HTTPException detail one level deeper than the auth middleware does.
    const detail = body?.detail;
    throw new Error(
      body?.error?.message ??
        detail?.error?.message ??
        (typeof detail === "string" ? detail : null) ??
        `${res.status} ${res.statusText}`,
    );
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  get: <T,>(p: string) => request<T>(p),
  post: <T,>(p: string, body?: unknown) =>
    request<T>(p, { method: "POST", body: JSON.stringify(body ?? {}) }),
  patch: <T,>(p: string, body: unknown) =>
    request<T>(p, { method: "PATCH", body: JSON.stringify(body) }),
  put: <T,>(p: string, body: unknown) =>
    request<T>(p, { method: "PUT", body: JSON.stringify(body) }),
  del: <T,>(p: string) => request<T>(p, { method: "DELETE" }),
};

/* ── view models ───────────────────────────────────────────────────────────── */

export type Health = {
  ok: boolean;
  season: string;
  players: number;
  fixtures: number;
  predictions: number;
  connectors: number;
  connectors_available: number;
  llm_configured: boolean;
  sqlite_vec: boolean;
  fpl_write_enabled: boolean;
};

export type Squad = {
  id: number;
  name: string;
  colour: string | null;
  fpl_entry_id: number | null;
  is_shadow: number;
  archived: number;
  projected_points?: number | null;
  settings?: Record<string, unknown>;
  state?: SquadState | null;
};

export type Pick = {
  player_id: number;
  /** The 1-15 squad slot, not GK/DEF/MID/FWD — that arrives as `position_name`. */
  position: number;
  is_captain: number;
  is_vice: number;
  purchase_price: number | null;
  selling_price: number | null;
  web_name?: string | null;
  position_name?: string | null;
  team_id?: number | null;
  team_short?: string | null;
};

export type SquadState = {
  id: number;
  gameweek: number;
  bank: number;
  squad_value: number;
  free_transfers: number;
  chip_active: string | null;
  picks: Pick[];
};

export type PlayerRef = {
  player_id: number;
  name: string;
  team_short: string | null;
  position: string | null;
  price: number | null;
  selling_price?: number | null;
  exp_points: number | null;
};

export type Recommendation = {
  id?: number;
  variant: string;
  gameweek: number;
  exp_points_gw: number;
  exp_points_horizon: number;
  sd_points_gw: number;
  hits_taken: number;
  chip_suggested: string | null;
  llm_rationale?: string | null;
  llm_critique?: unknown;
  accepted?: number | null;
  payload: {
    headline: string;
    gameweek: number;
    horizon: { gws: number; decay: number };
    recommendation: "act" | "do_nothing";
    transfers: { in: PlayerRef; out: PlayerRef; delta_exp_points_gw: number }[];
    hits: number;
    chip: string | null;
    squad: PlayerRef[];
    lineup: {
      xi: PlayerRef[];
      bench_order: PlayerRef[];
      captain: number | null;
      vice: number | null;
      formation: string;
    };
    totals: {
      exp_points_gw: number;
      sd_points_gw: number;
      exp_points_horizon: number;
      p_haul_captain: number | null;
      bank_after: number;
      free_transfers: number;
    };
    alternatives: { label: string; delta: number }[];
    future_gameweeks: {
      gameweek: number;
      transfers_in: PlayerRef[];
      transfers_out: PlayerRef[];
      hits: number;
      chip: string | null;
      exp_points: number;
    }[];
    chip_calendar: {
      chip: string;
      label: string;
      best_gw: number;
      gain: number;
      confidence: string;
      actionable: boolean;
      reason: string;
    }[];
    chip_warnings: { chip: string; label: string; message: string; severity: string }[];
    delta_vs_do_nothing: number | null;
  };
};

export type PlayerRow = {
  id: number;
  web_name: string;
  position: string;
  team_id: number;
  team_name: string;
  team_short: string;
  price: number | null;
  owned_pct: number | null;
  minutes: number | null;
  points: number | null;
  exp_points_gw: number | null;
  exp_points_horizon: number | null;
  p_start: number | null;
  p_haul: number | null;
  sd_points: number | null;
  value: number | null;
  form_sparkline: number[];
};

export type Deadline = {
  season_id: string;
  gameweek: number;
  deadline_utc: string | null;
  seconds_remaining: number | null;
  chips_available: string[];
};

export type SourceStatus = {
  id: string;
  category: string;
  enabled: boolean;
  available: boolean;
  unavailable_reason: string | null;
  requires_keys: string[];
  cadence: string;
  last_run: {
    started_at: string;
    status: string;
    docs_new: number;
    rows_upserted: number;
    error_text: string | null;
  } | null;
};

export type FeedItem = {
  type: "article" | "video" | "social" | "claim";
  id: number;
  title?: string | null;
  outlet?: string | null;
  url?: string | null;
  published_at?: string | null;
  excerpt?: string | null;
  seen_on_sites?: number;
  channel_title?: string | null;
  youtube_id?: string | null;
  view_count?: number | null;
  body_text?: string | null;
  author_handle?: string | null;
  platform?: string | null;
  retrieval_method?: string | null;
  player_name?: string | null;
  claim_type?: string | null;
  stance?: string | null;
  text_span?: string | null;
  source_id?: string | null;
};

/* ── hooks ─────────────────────────────────────────────────────────────────── */

const q = <T,>(key: unknown[], path: string, opts?: Partial<UseQueryOptions<T>>) =>
  useQuery<T>({ queryKey: key, queryFn: () => api.get<T>(path), ...opts });

export const useHealth = () => q<Health>(["health"], "/api/health");
export const useSquads = () => q<Squad[]>(["squads"], "/api/squads");
export const useSquad = (id?: number) =>
  q<Squad>(["squad", id], `/api/squads/${id}`, { enabled: !!id });
export const useDeadline = () =>
  q<Deadline>(["deadline"], "/api/gameweeks/current", { refetchInterval: 30_000 });
export const useSources = () => q<SourceStatus[]>(["sources"], "/api/sources");
export const useSettingsSchema = () => q<any>(["settings-schema"], "/api/settings/schema");
export const useGlobalSettings = () => q<any>(["settings-global"], "/api/settings/global");
export const usePundits = () => q<any>(["pundits"], "/api/pundits");
export const useModels = () => q<any[]>(["models"], "/api/models");
export const useJobs = () => q<any>(["jobs"], "/api/jobs");
export const useBacktests = () => q<any[]>(["backtests"], "/api/backtests");
export const useTicker = (gws = "1-8") =>
  q<any>(["ticker", gws], `/api/fixture-ticker?gws=${gws}`);
export const useFeed = (types: string, playerId?: number) =>
  q<FeedItem[]>(
    ["feed", types, playerId],
    `/api/feed?types=${types}${playerId ? `&player_id=${playerId}` : ""}`,
  );
export const usePlayer = (id?: number) =>
  q<any>(["player", id], `/api/players/${id}`, { enabled: !!id });
export const usePlayerFeatures = (id?: number) =>
  q<any[]>(["player-features", id], `/api/players/${id}/features`, { enabled: !!id });
export const usePlayerHistory = (id?: number) =>
  q<any[]>(["player-history", id], `/api/players/${id}/history`, { enabled: !!id });
export const usePlayerClaims = (id?: number) =>
  q<any[]>(["player-claims", id], `/api/players/${id}/claims`, { enabled: !!id });

export const usePlayers = (filters: Record<string, string | number | undefined> = {}) => {
  const qs = Object.entries(filters)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`)
    .join("&");
  return q<PlayerRow[]>(["players", qs], `/api/players${qs ? `?${qs}` : ""}`);
};

export const useRecommendations = (squadId?: number, gw?: number) =>
  q<Recommendation[]>(
    ["recs", squadId, gw],
    `/api/squads/${squadId}/recommendations${gw ? `?gw=${gw}` : ""}`,
    { enabled: !!squadId },
  );

export const useComparison = (ids: number[], gw?: number) =>
  q<any>(
    ["compare", ids.join(","), gw],
    `/api/squads/compare?ids=${ids.join(",")}${gw ? `&gw=${gw}` : ""}`,
    { enabled: ids.length > 1 },
  );

export function useRecommend(squadId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      gameweek?: number;
      force_refresh?: boolean;
      use_draft?: boolean;
    }) => api.post<Recommendation[]>(`/api/squads/${squadId}/recommend`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["recs", squadId] }),
  });
}

export function useWhatIf(squadId?: number) {
  return useMutation({
    mutationFn: (constraints: Record<string, unknown>) =>
      api.post<Recommendation>(`/api/squads/${squadId}/whatif`, { constraints }),
  });
}

/* ── working copy ("draft") ───────────────────────────────────────────────────
 * The set squad is what you actually own. A draft is a scratch copy you rearrange
 * freely — swap players, pull in a recommendation's picks, re-run the variants against
 * it — and nothing counts until you commit. */

export type Draft = Omit<SquadState, "picks"> & {
  source: string;
  picks: Pick[];
  ok: boolean;
  errors: string[];
};

export const useDraft = (squadId?: number) =>
  q<Draft>(["draft", squadId], `/api/squads/${squadId}/draft`, {
    enabled: !!squadId,
    retry: false, // a 404 just means "no working copy yet"
  });

export function useSeedDraft(squadId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { from_recommendation?: number } = {}) =>
      api.put<Draft>(`/api/squads/${squadId}/draft`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["draft", squadId] }),
  });
}

export function useEditDraft(squadId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      add?: number[];
      drop?: number[];
      captain?: number;
      vice?: number;
      bank?: number;
    }) => api.patch<Draft>(`/api/squads/${squadId}/draft`, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["draft", squadId] }),
  });
}

export function useDiscardDraft(squadId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.del(`/api/squads/${squadId}/draft`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["draft", squadId] }),
  });
}

export function useCommitDraft(squadId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post(`/api/squads/${squadId}/draft/commit`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["draft", squadId] });
      qc.invalidateQueries({ queryKey: ["squad", squadId] });
    },
  });
}

export function useSyncSquad(squadId?: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.post(`/api/squads/${squadId}/sync`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["squad", squadId] }),
  });
}

export function useAcceptRecommendation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.post(`/api/recommendations/${id}/accept`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["recs"] }),
  });
}

export function useRejectRecommendation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, reason }: { id: number; reason?: string }) =>
      api.post(`/api/recommendations/${id}/reject${reason ? `?reason=${encodeURIComponent(reason)}` : ""}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["recs"] }),
  });
}

export function useExplain() {
  return useMutation({
    mutationFn: (id: number) => api.post<{ rationale: string | null }>(`/api/recommendations/${id}/explain`),
  });
}

export function useCritique() {
  return useMutation({
    mutationFn: (id: number) => api.post<{ critique: any }>(`/api/recommendations/${id}/critique`),
  });
}

export function usePatchSettings(scope: "global" | number) {
  const qc = useQueryClient();
  const path = scope === "global" ? "/api/settings/global" : `/api/settings/squad/${scope}`;
  return useMutation({
    mutationFn: (values: Record<string, unknown>) => api.patch(path, { values }),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useCreateSquad() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; fpl_entry_id?: number; clone_from?: number }) =>
      api.post<Squad>("/api/squads", body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["squads"] }),
  });
}

export function useDeleteSquad() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.del(`/api/squads/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["squads"] }),
  });
}

export function useRunJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => api.post(`/api/jobs/${name}/run`),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export type KeyCheck = { key: string; service: string; ok: boolean; detail: string };

export function useVerifyKeys() {
  return useMutation({
    mutationFn: () => api.post<{ results: KeyCheck[]; failed: number }>("/api/settings/verify"),
  });
}

/* ── formatting ────────────────────────────────────────────────────────────── */

export const money = (tenths: number | null | undefined) =>
  tenths == null ? "—" : `£${(tenths / 10).toFixed(1)}m`;

export const pts = (v: number | null | undefined, dp = 1) =>
  v == null ? "—" : v.toFixed(dp);

export const pct = (v: number | null | undefined, dp = 0) =>
  v == null ? "—" : `${(v * 100).toFixed(dp)}%`;

export function countdown(seconds: number | null | undefined) {
  if (seconds == null) return "—";
  if (seconds <= 0) return "deadline passed";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

/** Amber at 24h, red at 2h. */
export function deadlineTone(seconds: number | null | undefined) {
  if (seconds == null) return "text-muted";
  if (seconds <= 7200) return "text-neg";
  if (seconds <= 86400) return "text-warn";
  return "text-slate-200";
}
