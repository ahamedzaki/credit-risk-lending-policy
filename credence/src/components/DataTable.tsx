import { useMemo, useState } from "react";
import { Icon } from "./Icon";

export interface Column<T> {
  key: string;
  header: string;
  num?: boolean;
  render: (row: T) => React.ReactNode;
  sortValue?: (row: T) => number | string;
}

export function DataTable<T extends { loan_id?: string; id?: string }>({
  rows,
  columns,
  search,
  searchKeys,
  onRowClick,
  initialSort,
}: {
  rows: T[];
  columns: Column<T>[];
  search?: string;
  searchKeys?: (row: T) => string;
  onRowClick?: (row: T) => void;
  initialSort?: { key: string; dir: "asc" | "desc" };
}) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState(initialSort ?? { key: columns[0].key, dir: "asc" as const });
  const query = (search ?? q).trim().toLowerCase();

  const view = useMemo(() => {
    let r = rows;
    if (query && searchKeys) r = r.filter((row) => searchKeys(row).toLowerCase().includes(query));
    const col = columns.find((c) => c.key === sort.key);
    if (col?.sortValue) {
      const sv = col.sortValue;
      r = [...r].sort((a, b) => {
        const va = sv(a);
        const vb = sv(b);
        const cmp = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb));
        return sort.dir === "asc" ? cmp : -cmp;
      });
    }
    return r;
  }, [rows, columns, query, searchKeys, sort]);

  function toggleSort(key: string) {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" }));
  }

  return (
    <div className="tablewrap">
      {search === undefined && (
        <div className="tabletools">
          <span className="searchbox">
            <Icon name="search" size={15} />
            <input
              placeholder="Search borrower ID, purpose, decision…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              aria-label="Search borrowers"
            />
          </span>
          <span className="tablecount">
            {view.length.toLocaleString("en-US")} of {rows.length.toLocaleString("en-US")}
          </span>
        </div>
      )}
      <div className="tablescroll">
        <table className="data">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  className={c.num ? "num" : undefined}
                  aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}
                  onClick={() => c.sortValue && toggleSort(c.key)}
                  style={{ cursor: c.sortValue ? "pointer" : "default" }}
                >
                  {c.header}
                  {c.sortValue && (
                    <span className="caret">
                      <Icon name={sort.key === c.key && sort.dir === "asc" ? "arrowUp" : "arrowDown"} size={11} strokeWidth={2} />
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.map((row, i) => (
              <tr
                key={row.loan_id ?? row.id ?? i}
                onClick={() => onRowClick?.(row)}
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && onRowClick?.(row)}
              >
                {columns.map((c) => (
                  <td key={c.key} className={c.num ? "num" : undefined}>
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
