import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Shell } from "./components/Chrome";
import Dashboard from "./routes/Dashboard";
import { PlayerDetail, PlayerExplorer } from "./routes/Players";
import {
  Chat,
  Compare,
  Feed,
  More,
  Performance,
  Planner,
  Settings,
  SquadView,
  Ticker,
} from "./routes/Screens";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1, refetchOnWindowFocus: false },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Shell>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/squad" element={<SquadView />} />
            <Route path="/players" element={<PlayerExplorer />} />
            <Route path="/players/:id" element={<PlayerDetail />} />
            <Route path="/planner" element={<Planner />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/ticker" element={<Ticker />} />
            <Route path="/feed" element={<Feed />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/more" element={<More />} />
          </Routes>
        </Shell>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
