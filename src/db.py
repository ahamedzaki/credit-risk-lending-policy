"""Thin DuckDB helpers: open the project DB, run a .sql file with ${...} substitution."""
from __future__ import annotations

import pathlib
import re
import textwrap

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"


def connect(db_path: str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path, read_only=read_only)


def _render(sql_text: str, params: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in params:
            raise KeyError(f"SQL placeholder ${{{key}}} has no value in sql_params()")
        return str(params[key])

    return re.sub(r"\$\{(\w+)\}", repl, sql_text)


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str, params: dict[str, str]) -> list:
    """Execute every statement in sql/<name>. Returns the result of the LAST statement
    (as a list of tuples) so the runner can print a small summary.

    Line comments (``-- ...``) are stripped BEFORE splitting on ``;`` so a semicolon
    inside a comment cannot break the split. String literals in these files never
    contain ``--`` or ``;``; keep it that way.
    """
    path = SQL_DIR / name
    rendered = _render(path.read_text(), params)
    no_comments = "\n".join(re.sub(r"--.*$", "", line) for line in rendered.splitlines())
    statements = [s.strip() for s in no_comments.split(";") if s.strip()]
    last = []
    for stmt in statements:
        last = con.execute(stmt).fetchall()
    return last


def show(rows: list, headers: list[str] | None = None) -> None:
    if not rows:
        print("   (no rows)")
        return
    for r in rows[:25]:
        print("   " + " | ".join(str(x) for x in r))
    if len(rows) > 25:
        print(f"   ... (+{len(rows) - 25} more)")
