import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { Icon, type IconName } from "./Icon";

const PRIMARY: { to: string; label: string; icon: IconName }[] = [
  { to: "/", label: "Overview", icon: "overview" },
  { to: "/portfolio", label: "Portfolio", icon: "portfolio" },
  { to: "/borrowers", label: "Borrowers", icon: "borrowers" },
  { to: "/decisions", label: "Decisions", icon: "decisions" },
  { to: "/monitoring", label: "Monitoring", icon: "monitoring" },
];

const FOOTER: { to: string; label: string; icon: IconName }[] = [
  { to: "/settings", label: "Settings", icon: "settings" },
  { to: "/data-model", label: "Data / Model", icon: "info" },
];

function Item({ to, label, icon }: { to: string; label: string; icon: IconName }) {
  return (
    <NavLink to={to} end={to === "/"} className="navitem">
      <Icon name={icon} size={19} />
      <span>{label}</span>
    </NavLink>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="shell">
      <nav className="rail" aria-label="Primary">
        <div className="rail__mark">
          <b>CREDENCE</b>
          <span>Risk Intelligence</span>
        </div>
        <div className="rail__group">
          {PRIMARY.map((i) => (
            <Item key={i.to} {...i} />
          ))}
        </div>
        <div className="rail__spacer" />
        <div className="rail__group">
          {FOOTER.map((i) => (
            <Item key={i.to} {...i} />
          ))}
        </div>
      </nav>
      <main className="main">{children}</main>
    </div>
  );
}

export function PageHead({
  title,
  lede,
  aside,
}: {
  title: string;
  lede?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="pagehead">
      <div>
        <h1>{title}</h1>
        {lede && <p>{lede}</p>}
      </div>
      {aside && <div className="pagehead__aside">{aside}</div>}
    </div>
  );
}
