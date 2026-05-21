"""Convert the LSApp TSV to a RecBole atomic `.inter` file.

LSApp (https://github.com/aliannejadi/LSApp) is an Android sequential app-usage
dataset. Schema of `lsapp.tsv`:

    user_id      int
    session_id   int
    timestamp    ISO-8601 (UTC, naive)
    app_name     string
    event_type   "Opened" | "Closed"

Only `Opened` rows are real foreground events — Closed rows are bookends. We
keep Opened only and write user/app/timestamp in RecBole atomic format.

Output: <out>/<name>.inter with columns
    user_id:token  item_id:token  timestamp:float
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MIN_INTERACTIONS_PER_USER = 5
MIN_INTERACTIONS_PER_APP = 5


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    expected = {"user_id", "timestamp", "app_name", "event_type"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing columns {missing}")
    df = df[df["event_type"] == "Opened"].copy()
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce").astype("int64") // 10**9
    )
    df = df.dropna(subset=["timestamp"])
    return df.rename(columns={"app_name": "item_id"})[["user_id", "item_id", "timestamp"]]


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
    ap.add_argument("--tsv", required=True, type=Path, help="path to lsapp.tsv")
    ap.add_argument("--out", required=True, type=Path, help="output dir")
    ap.add_argument("--name", default="lsapp")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"not found: {args.tsv}")

    df = load(args.tsv)
    df = df.sort_values(["user_id", "timestamp"])
    df = filter_sparse(df)

    out_path = write_inter(df, args.out, args.name)
    print(f"wrote {out_path}")
    print(f"  rows:  {len(df):>8}")
    print(f"  users: {df['user_id'].nunique():>8}")
    print(f"  apps:  {df['item_id'].nunique():>8}")


if __name__ == "__main__":
    main()
