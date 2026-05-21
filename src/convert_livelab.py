"""Convert LiveLab `appusage.sql` to a RecBole atomic `.inter` file.

LiveLab ships SQL dumps. The dataset we want is `appusage.sql` (inside
`apps.tar.gz` from https://yecl.org/livelab/traces.html), schema:

    id   integer
    uid  integer    -- participant
    name text       -- app name
    time integer    -- POSIX seconds (foreground start)
    duration integer

We parse `INSERT INTO ... VALUES (...), (...), ...;` statements directly so
this works regardless of whether the dump is MySQL- or SQLite-flavored.

Output: <out>/<name>.inter with columns
    user_id:token  item_id:token  timestamp:float
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

MIN_INTERACTIONS_PER_USER = 5
MIN_INTERACTIONS_PER_APP = 5

# Matches one `(...)` tuple inside a VALUES list. Handles single-quoted strings
# (with escaped quotes), NULLs, and numeric literals.
_TUPLE_RE = re.compile(
    r"\(\s*"
    r"(?P<id>\d+)\s*,\s*"
    r"'(?P<uid>(?:[^'\\]|\\.|'')*)'\s*,\s*"
    r"'(?P<name>(?:[^'\\]|\\.|'')*)'\s*,\s*"
    r"(?P<time>\d+)\s*,\s*"
    r"(?P<duration>\d+|NULL)\s*"
    r"\)",
    re.IGNORECASE,
)

# Finds the VALUES payload of INSERT INTO appusage statements.
_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+[`\"']?appusage[`\"']?\s*(?:\([^)]*\))?\s*VALUES\s*",
    re.IGNORECASE,
)


def parse_appusage_sql(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict] = []
    for m in _INSERT_RE.finditer(text):
        # Scan tuples until the terminating semicolon for this INSERT.
        end = text.find(";", m.end())
        if end == -1:
            end = len(text)
        chunk = text[m.end():end]
        for tup in _TUPLE_RE.finditer(chunk):
            rows.append({
                "user_id": tup.group("uid"),
                "item_id": tup.group("name").replace("''", "'").replace("\\'", "'"),
                "timestamp": float(tup.group("time")),
            })
    if not rows:
        sys.exit(
            f"parsed 0 rows from {path}. Open the file and check it really is an "
            "`appusage` SQL dump with INSERT statements."
        )
    return pd.DataFrame(rows)


def filter_sparse(df: pd.DataFrame) -> pd.DataFrame:
    while True:
        before = len(df)
        uc = df.groupby("user_id").size()
        df = df[df["user_id"].isin(uc[uc >= MIN_INTERACTIONS_PER_USER].index)]
        ic = df.groupby("item_id").size()
        df = df[df["item_id"].isin(ic[ic >= MIN_INTERACTIONS_PER_APP].index)]
        if len(df) == before:
            return df


def write_inter(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.inter"
    with path.open("w") as fh:
        fh.write("user_id:token\titem_id:token\ttimestamp:float\n")
        df.to_csv(fh, sep="\t", header=False, index=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sql", required=True, type=Path,
        help="path to appusage.sql (extract apps.tar.gz first)",
    )
    ap.add_argument("--out", required=True, type=Path, help="output dir")
    ap.add_argument("--name", default="fitphone")
    ap.add_argument(
        "--drop-springboard", action="store_true",
        help="drop iOS SpringBoard (home screen) events — usually noise for next-app prediction",
    )
    args = ap.parse_args()

    if not args.sql.exists():
        sys.exit(f"not found: {args.sql}")

    df = parse_appusage_sql(args.sql)
    if args.drop_springboard:
        df = df[df["item_id"] != "SpringBoard"]
    df = df.dropna().sort_values(["user_id", "timestamp"])
    df = filter_sparse(df)

    out_path = write_inter(df[["user_id", "item_id", "timestamp"]], args.out, args.name)

    print(f"wrote {out_path}")
    print(f"  rows:  {len(df):>8}")
    print(f"  users: {df['user_id'].nunique():>8}")
    print(f"  apps:  {df['item_id'].nunique():>8}")


if __name__ == "__main__":
    main()
