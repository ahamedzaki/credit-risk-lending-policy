import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { DEFAULT_CUTS } from "../lib/risk";

interface SettingsState {
  density: "comfortable" | "compact";
  reducedMotion: boolean;
  cuts: { medium: number; high: number; critical: number };
  setDensity: (d: "comfortable" | "compact") => void;
  setReducedMotion: (b: boolean) => void;
  setCut: (k: "medium" | "high" | "critical", v: number) => void;
}

const Ctx = createContext<SettingsState | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [density, setDensity] = useState<"comfortable" | "compact">("comfortable");
  const [reducedMotion, setReducedMotion] = useState(false);
  const [cuts, setCuts] = useState(DEFAULT_CUTS);

  const value = useMemo<SettingsState>(
    () => ({
      density,
      reducedMotion,
      cuts,
      setDensity,
      setReducedMotion,
      setCut: (k, v) => setCuts((c) => ({ ...c, [k]: v })),
    }),
    [density, reducedMotion, cuts],
  );

  return (
    <Ctx.Provider value={value}>
      <div
        data-density={density}
        style={reducedMotion ? ({ "--t-control": "0ms", "--t-fade": "0ms", "--t-drawer": "0ms" } as React.CSSProperties) : undefined}
      >
        {children}
      </div>
    </Ctx.Provider>
  );
}

export function useSettings() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useSettings outside provider");
  return c;
}
