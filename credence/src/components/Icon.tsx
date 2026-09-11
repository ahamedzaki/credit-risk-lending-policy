/** One consistent icon set: 1.5px stroke, round caps, 24-unit grid, sized via prop. */
import type { CSSProperties } from "react";

type Name =
  | "overview"
  | "portfolio"
  | "borrowers"
  | "decisions"
  | "monitoring"
  | "settings"
  | "info"
  | "search"
  | "chevronRight"
  | "chevronDown"
  | "arrowUp"
  | "arrowDown"
  | "close"
  | "external"
  | "filter"
  | "check"
  | "dot";

const P: Record<Name, string> = {
  overview: "M4 13h6V4H4v9Zm0 7h6v-4H4v4Zm10 0h6V11h-6v9Zm0-16v4h6V4h-6Z",
  portfolio: "M4 19V5m0 14h16M8 15l3.5-4 3 2.5L20 8",
  borrowers: "M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm7 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM3 20v-1a5 5 0 0 1 5-5h2a5 5 0 0 1 5 5v1M15 14h1a5 5 0 0 1 5 5v1",
  decisions: "M12 3v18M5 8l7-5 7 5M6 11v6a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-6",
  monitoring: "M3 12h4l3-7 4 14 3-7h4",
  settings:
    "M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm8.5-3.5-1.8-.4a6.8 6.8 0 0 0-.6-1.5l1-1.5-1.8-1.8-1.5 1a6.8 6.8 0 0 0-1.5-.6L13.6 3.5h-3.2l-.4 1.8a6.8 6.8 0 0 0-1.5.6l-1.5-1-1.8 1.8 1 1.5c-.3.5-.5 1-.6 1.5l-1.8.4v3.2l1.8.4c.1.5.3 1 .6 1.5l-1 1.5 1.8 1.8 1.5-1c.5.3 1 .5 1.5.6l.4 1.8h3.2l.4-1.8c.5-.1 1-.3 1.5-.6l1.5 1 1.8-1.8-1-1.5c.3-.5.5-1 .6-1.5l1.8-.4v-3.2Z",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-13h.01M11 11h1v5h1",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm5-2 5 5",
  chevronRight: "m9 6 6 6-6 6",
  chevronDown: "m6 9 6 6 6-6",
  arrowUp: "M12 19V5m0 0-6 6m6-6 6 6",
  arrowDown: "M12 5v14m0 0 6-6m-6 6-6-6",
  close: "m6 6 12 12M18 6 6 18",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  filter: "M4 5h16l-6 8v6l-4 2v-8L4 5Z",
  check: "m5 12 4.5 4.5L19 7",
  dot: "M12 12h.01",
};

export function Icon({
  name,
  size = 20,
  style,
  strokeWidth = 1.6,
}: {
  name: Name;
  size?: number;
  style?: CSSProperties;
  strokeWidth?: number;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={style}
    >
      <path d={P[name]} />
    </svg>
  );
}

export type { Name as IconName };
