"""Habits / goals domain. Tracks self-reported mindful actions and streaks.

Unlike the event log (passive app usage), habits are user-affirmed actions:
taking a mindful break, answering a reflection, staying off the phone at night.
We seed a default set per user on first access so the UI always has something.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import List

DEFAULT_HABITS = [
    ("mindful_break", "Mindful breaks", "☕", 7),
    ("no_phone_before_bed", "No phone before bed", "\U0001f319", 7),
    ("morning_no_scroll", "Morning without scrolling", "☀️", 7),
    ("intentional_check", "Intentional phone checks", "\U0001f3af", 7),
]


def init_habits_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS habits ("
        "user_id TEXT NOT NULL,"
        "key TEXT NOT NULL,"
        "name TEXT NOT NULL,"
        "icon TEXT NOT NULL,"
        "target_per_week INTEGER NOT NULL,"
        "PRIMARY KEY (user_id, key)"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS habit_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "user_id TEXT NOT NULL,"
        "habit_key TEXT NOT NULL,"
        "timestamp REAL NOT NULL"
        ")"
    )


def ensure_defaults(conn: sqlite3.Connection, user_id: str) -> None:
    existing = conn.execute(
        "SELECT 1 FROM habits WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()
    if existing:
        return
    conn.executemany(
        "INSERT INTO habits (user_id, key, name, icon, target_per_week) VALUES (?, ?, ?, ?, ?)",
        [(user_id, k, n, i, t) for (k, n, i, t) in DEFAULT_HABITS],
    )


def log_habit(conn: sqlite3.Connection, user_id: str, habit_key: str, ts: float) -> None:
    conn.execute(
        "INSERT INTO habit_events (user_id, habit_key, timestamp) VALUES (?, ?, ?)",
        (user_id, habit_key, ts),
    )


def _event_dates(conn: sqlite3.Connection, user_id: str) -> set:
    rows = conn.execute(
        "SELECT timestamp FROM habit_events WHERE user_id = ?", (user_id,)
    ).fetchall()
    return {datetime.fromtimestamp(t).date() for (t,) in rows}


def _streak(dates: set, today) -> int:
    """Consecutive days ending today (or yesterday) with at least one habit event."""
    if not dates:
        return 0
    start = today if today in dates else today - timedelta(days=1)
    if start not in dates:
        return 0
    streak = 0
    day = start
    while day in dates:
        streak += 1
        day -= timedelta(days=1)
    return streak


def get_habits(conn: sqlite3.Connection, user_id: str) -> dict:
    ensure_defaults(conn, user_id)
    now = datetime.now()
    today = now.date()
    week_ago = (now - timedelta(days=7)).timestamp()

    defs = conn.execute(
        "SELECT key, name, icon, target_per_week FROM habits WHERE user_id = ?",
        (user_id,),
    ).fetchall()

    week_counts = dict(
        conn.execute(
            "SELECT habit_key, COUNT(*) FROM habit_events "
            "WHERE user_id = ? AND timestamp >= ? GROUP BY habit_key",
            (user_id, week_ago),
        ).fetchall()
    )
    today_counts = dict(
        conn.execute(
            "SELECT habit_key, COUNT(*) FROM habit_events "
            "WHERE user_id = ? AND timestamp >= ? GROUP BY habit_key",
            (user_id, datetime(today.year, today.month, today.day).timestamp()),
        ).fetchall()
    )

    habits = [
        {
            "key": k,
            "name": n,
            "icon": i,
            "target_per_week": t,
            "this_week": week_counts.get(k, 0),
            "today": today_counts.get(k, 0),
        }
        for (k, n, i, t) in defs
    ]

    dates = _event_dates(conn, user_id)
    week_dots: List[dict] = []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        week_dots.append({"label": "MTWTFSS"[day.weekday()], "done": day in dates})

    return {
        "streak": _streak(dates, today),
        "habits": habits,
        "week_dots": week_dots,
        "mindful_breaks_today": today_counts.get("mindful_break", 0),
        "mindful_breaks_target": 7,
    }
