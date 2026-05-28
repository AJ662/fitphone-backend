"""Usage analytics derived from the raw event log.

LSApp logs Opened/Closed bookends that fire within the same second, so pairing
them gives near-zero durations. Instead we estimate *foreground* time the
standard way: an opened app stays in the foreground until the next event in the
same session, capped at the session-gap so an idle phone doesn't count.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

DWELL_CAP_SECONDS = 5 * 60  # don't credit more than this to a single open
TAIL_DWELL_SECONDS = 30  # last open in a session gets a small fixed credit
NIGHT_START_HOUR = 22  # 22:00
NIGHT_END_HOUR = 6  # 06:00


def _is_night(dt: datetime) -> bool:
    return dt.hour >= NIGHT_START_HOUR or dt.hour < NIGHT_END_HOUR


def _dwells(rows: List[tuple]) -> List[tuple]:
    """rows: (session_id, app, ts) sorted by ts. Returns (app, ts, dwell_seconds)."""
    by_session: Dict[int, List[tuple]] = defaultdict(list)
    for session_id, app, ts in rows:
        by_session[session_id].append((app, ts))
    out = []
    for events in by_session.values():
        for i, (app, ts) in enumerate(events):
            if i + 1 < len(events):
                dwell = min(events[i + 1][1] - ts, DWELL_CAP_SECONDS)
            else:
                dwell = TAIL_DWELL_SECONDS
            out.append((app, ts, max(dwell, 0)))
    return out


def compute_stats(conn: sqlite3.Connection, user_id: str, social_apps: set) -> dict:
    rows = conn.execute(
        "SELECT session_id, app, timestamp FROM events "
        "WHERE user_id = ? AND event_type = 'Opened' ORDER BY timestamp",
        (user_id,),
    ).fetchall()

    if not rows:
        return {
            "has_data": False,
            "screen_time_seconds": 0,
            "session_count": 0,
            "top_apps": [],
            "hourly_seconds": [0] * 24,
            "continuous_social_seconds": 0,
            "night_minutes_this_week": 0,
            "night_minutes_last_week": 0,
            "night_change_pct": None,
        }

    now = datetime.now()
    today = now.date()

    today_rows = [(s, a, t) for (s, a, t) in rows if datetime.fromtimestamp(t).date() == today]
    dwells = _dwells(today_rows)

    screen_time = sum(d for _, _, d in dwells)
    per_app: Dict[str, float] = defaultdict(float)
    hourly = [0.0] * 24
    for app, ts, dwell in dwells:
        per_app[app] += dwell
        hourly[datetime.fromtimestamp(ts).hour] += dwell

    top_apps = sorted(
        ({"app": a, "seconds": int(s)} for a, s in per_app.items()),
        key=lambda x: x["seconds"],
        reverse=True,
    )[:6]

    # Longest consecutive run of social-category opens within a single session.
    best_social = 0.0
    by_session: Dict[int, List[tuple]] = defaultdict(list)
    for s, a, t in today_rows:
        by_session[s].append((a, t))
    for events in by_session.values():
        run_start = None
        prev_ts = None
        for app, ts in events:
            if app in social_apps:
                if run_start is None:
                    run_start = ts
                prev_ts = ts
            else:
                if run_start is not None:
                    best_social = max(best_social, prev_ts - run_start)
                run_start = None
        if run_start is not None and prev_ts is not None:
            best_social = max(best_social, prev_ts - run_start)

    # Night use: this week vs previous week (foreground minutes in night window).
    week_ago = (now - timedelta(days=7)).timestamp()
    two_weeks_ago = (now - timedelta(days=14)).timestamp()
    recent_rows = [(s, a, t) for (s, a, t) in rows if t >= two_weeks_ago]
    night_this = night_last = 0.0
    for app, ts, dwell in _dwells(recent_rows):
        if not _is_night(datetime.fromtimestamp(ts)):
            continue
        if ts >= week_ago:
            night_this += dwell
        else:
            night_last += dwell
    night_change = None
    if night_last > 0:
        night_change = round((night_this - night_last) / night_last * 100)

    return {
        "has_data": True,
        "screen_time_seconds": int(screen_time),
        "session_count": len({s for s, _, _ in today_rows}),
        "top_apps": top_apps,
        "hourly_seconds": [int(h) for h in hourly],
        "continuous_social_seconds": int(best_social),
        "night_minutes_this_week": int(night_this // 60),
        "night_minutes_last_week": int(night_last // 60),
        "night_change_pct": night_change,
    }
