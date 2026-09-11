import { PageHead } from "../components/AppShell";
import { Section, Seg, Switch } from "../components/primitives";
import { useSettings } from "../app/settings";
import { pct } from "../lib/format";

export function Settings() {
  const s = useSettings();
  return (
    <div className="page">
      <PageHead title="Settings" lede="Display and risk-band preferences for this session." />

      <Section title="Display" desc="">
        <div className="setting">
          <div>
            <h3>Density</h3>
            <p>Comfortable spacing for reading, or compact to fit more on screen.</p>
          </div>
          <Seg options={["comfortable", "compact"] as const} value={s.density} onChange={s.setDensity} />
        </div>
        <div className="setting">
          <div>
            <h3>Reduce motion</h3>
            <p>Disable the drawer slide and chart transitions. Also follows your system setting.</p>
          </div>
          <Switch on={s.reducedMotion} onToggle={() => s.setReducedMotion(!s.reducedMotion)} label="Reduce motion" />
        </div>
      </Section>

      <Section title="Risk-band cut-offs" desc="Probability-of-default boundaries used to place loans on the risk spectrum.">
        {(["medium", "high", "critical"] as const).map((k) => (
          <div className="setting" key={k}>
            <div>
              <h3>{k === "medium" ? "Low → Medium" : k === "high" ? "Medium → High" : "High → Critical"}</h3>
              <p>Loans move into the higher band at PD {pct(s.cuts[k], 0)}.</p>
            </div>
            <input
              type="range"
              min={k === "medium" ? 0.02 : k === "high" ? 0.06 : 0.14}
              max={k === "medium" ? 0.08 : k === "high" ? 0.16 : 0.3}
              step={0.01}
              value={s.cuts[k]}
              onChange={(e) => s.setCut(k, Number(e.target.value))}
              className="sim__slider"
              style={{ width: 220, margin: 0 }}
              aria-label={`${k} cut-off`}
            />
          </div>
        ))}
        <p className="note">
          Default cut-offs are 5% / 10% / 20%, matching the pipeline's risk-band definitions.
          Changing them here re-labels the borrower table and risk pills — it's a display
          classification, separate from the underwriting approval cut-off (15%, the real
          backtested policy) explored in Decisions. Moving one doesn't move the other.
        </p>
      </Section>

      <Section title="Currency" desc="">
        <div className="setting">
          <div>
            <h3>USD</h3>
            <p>
              All figures are in US dollars — the unit of the underlying Lending Club portfolio
              and pipeline. This is fixed for the demo.
            </p>
          </div>
        </div>
      </Section>
    </div>
  );
}
