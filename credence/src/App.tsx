import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Overview } from "./pages/Overview";
import { Portfolio } from "./pages/Portfolio";
import { Borrowers } from "./pages/Borrowers";
import { Decisions } from "./pages/Decisions";
import { Monitoring } from "./pages/Monitoring";
import { DataModel } from "./pages/DataModel";
import { Settings } from "./pages/Settings";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/borrowers" element={<Borrowers />} />
        <Route path="/decisions" element={<Decisions />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/data-model" element={<DataModel />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="*" element={<Overview />} />
      </Routes>
    </AppShell>
  );
}
