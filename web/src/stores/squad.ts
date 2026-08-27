import { create } from "zustand";
import { persist } from "zustand/middleware";

type Variant = "safe" | "balanced" | "aggressive";

type SquadStore = {
  activeSquadId: number | null;
  comparisonSet: number[];
  variant: Variant;
  watchlist: number[];
  setActive: (id: number) => void;
  toggleComparison: (id: number) => void;
  clearComparison: () => void;
  setVariant: (v: Variant) => void;
  toggleWatch: (id: number) => void;
};

export const useSquadStore = create<SquadStore>()(
  persist(
    (set) => ({
      activeSquadId: null,
      comparisonSet: [],
      variant: "balanced",
      watchlist: [],
      setActive: (id) => set({ activeSquadId: id }),
      toggleComparison: (id) =>
        set((s) => ({
          comparisonSet: s.comparisonSet.includes(id)
            ? s.comparisonSet.filter((x) => x !== id)
            : [...s.comparisonSet, id],
        })),
      clearComparison: () => set({ comparisonSet: [] }),
      setVariant: (variant) => set({ variant }),
      toggleWatch: (id) =>
        set((s) => ({
          watchlist: s.watchlist.includes(id)
            ? s.watchlist.filter((x) => x !== id)
            : [...s.watchlist, id],
        })),
    }),
    { name: "fplai.squad" },
  ),
);
