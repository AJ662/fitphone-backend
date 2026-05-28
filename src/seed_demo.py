"""Seed events.db with a realistic demo user copied from LSApp.

Takes one LSApp user's last N days of Opened/Closed events and shifts the
timestamps forward so the most recent day lands on today. This gives the
frontend's /stats and /habits endpoints real-looking data out of the box,
without waiting for the Simulate page to accumulate history.

    uv run python -m src.seed_demo --tsv data/raw/lsapp.tsv --reset
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd

from .api import DB_PATH, init_db
from .habits import init_habits_db, log_habit

KEEP_EVENTS = {"Opened", "Closed"}


def load_slice(tsv: Path, lsapp_user: int, days: int) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    df = df[(df["user_id"] == lsapp_user) & (df["event_type"].isin(KEEP_EVENTS))].copy()
    if df.empty:
        sys.exit(f"no Opened/Closed events for LSApp user {lsapp_user}")
    df["dt"] = pd.to_datetime(df["timestamp"])
    last_days = sorted(df["dt"].dt.date.unique())[-days:]
    df = df[df["dt"].dt.date.isin(last_days)].sort_values("dt")
    return df


def shift_to_today(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["dt"].dt.date.max()
    delta = datetime.now().date() - max_date
    df = df.copy()
    df["dt"] = df["dt"] + pd.Timedelta(days=delta.days)
    df["ts"] = (df["dt"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return df


def seed_events(conn: sqlite3.Connection, demo_user: str, df: pd.DataFrame, reset: bool) -> int:
    if reset:
        conn.execute("DELETE FROM events WHERE user_id = ?", (demo_user,))
    conn.executemany(
        "INSERT INTO events (user_id, session_id, app, event_type, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (demo_user, int(r.session_id), r.app_name, r.event_type, float(r.ts))
            for r in df.itertuples()
        ],
    )
    return len(df)


def seed_habits(conn: sqlite3.Connection, demo_user: str, days: int, reset: bool) -> int:
    """Affirm a couple of mindful actions on most recent days so a streak shows."""
    if reset:
        conn.execute("DELETE FROM habit_events WHERE user_id = ?", (demo_user,))
    today = datetime.now().date()
    n = 0
    for offset in range(days):
        day = today - timedelta(days=offset)
        # skip one day a few back so the streak is a realistic length, not "perfect"
        if offset == days - 2:
            continue
        for key, hour in [("mindful_break", 11), ("intentional_check", 15), ("mindful_break", 19)]:
            ts = datetime.combine(day, time(hour, 0)).timestamp()
            log_habit(conn, demo_user, key, ts)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, type=Path)
    ap.add_argument("--lsapp-user", type=int, default=116)
    ap.add_argument("--demo-user", default="demo")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--reset", action="store_true", help="delete existing demo rows first")
    args = ap.parse_args()

    if not args.tsv.exists():
        sys.exit(f"not found: {args.tsv}")

    df = shift_to_today(load_slice(args.tsv, args.lsapp_user, args.days))

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        init_habits_db(conn)
        n_events = seed_events(conn, args.demo_user, df, args.reset)
        n_habits = seed_habits(conn, args.demo_user, args.days, args.reset)

    print(f"seeded user '{args.demo_user}' from LSApp user {args.lsapp_user}")
    print(f"  events: {n_events} across {df['dt'].dt.date.nunique()} days (latest = today)")
    print(f"  habit events: {n_habits}")


if __name__ == "__main__":
    main()
