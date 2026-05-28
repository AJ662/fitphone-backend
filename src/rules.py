"""User-configurable nudge rules + the predict-and-act decision.

A rule says: "when the model predicts an app in category X, fire this action"
— show an image and/or a notification with this message — but no more than
once per cooldown window. /next_action is polled continuously by the client;
it predicts the next app, finds a matching enabled rule, and returns the action
to perform (or null).
"""

from __future__ import annotations

import random
import sqlite3
import time
from typing import List, Optional

DEFAULT_COOLDOWN = 60  # seconds
RECENT_HISTORY = 50


def init_rules_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nudge_rules ("
        "user_id TEXT NOT NULL,"
        "category TEXT NOT NULL,"
        "enabled INTEGER NOT NULL DEFAULT 1,"
        "message TEXT NOT NULL DEFAULT '',"
        "image TEXT NOT NULL DEFAULT '',"
        "notify INTEGER NOT NULL DEFAULT 1,"
        "show_image INTEGER NOT NULL DEFAULT 1,"
        "cooldown_seconds INTEGER NOT NULL DEFAULT 60,"
        "last_fired REAL NOT NULL DEFAULT 0,"
        "PRIMARY KEY (user_id, category)"
        ")"
    )


def ensure_defaults(conn: sqlite3.Connection, user_id: str, nudges) -> None:
    existing = conn.execute(
        "SELECT 1 FROM nudge_rules WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()
    if existing:
        return
    rows = []
    for category, body in nudges.categories.items():
        rows.append((user_id, category, 1, "", body.get("image", ""), 1, 1, DEFAULT_COOLDOWN, 0.0))
    conn.executemany(
        "INSERT INTO nudge_rules "
        "(user_id, category, enabled, message, image, notify, show_image, cooldown_seconds, last_fired) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _row_to_rule(row) -> dict:
    cat, enabled, message, image, notify, show_image, cooldown, last_fired = row
    return {
        "category": cat,
        "enabled": bool(enabled),
        "message": message,
        "image": image,
        "notify": bool(notify),
        "show_image": bool(show_image),
        "cooldown_seconds": cooldown,
        "last_fired": last_fired,
    }


def get_rules(conn: sqlite3.Connection, user_id: str, nudges) -> List[dict]:
    ensure_defaults(conn, user_id, nudges)
    rows = conn.execute(
        "SELECT category, enabled, message, image, notify, show_image, cooldown_seconds, last_fired "
        "FROM nudge_rules WHERE user_id = ? ORDER BY category",
        (user_id,),
    ).fetchall()
    return [_row_to_rule(r) for r in rows]


def update_rule(conn: sqlite3.Connection, user_id: str, category: str, fields: dict) -> None:
    allowed = {"enabled", "message", "notify", "show_image", "cooldown_seconds"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return
    for k in ("enabled", "notify", "show_image"):
        if k in sets:
            sets[k] = int(bool(sets[k]))
    cols = ", ".join(f"{k} = ?" for k in sets)
    conn.execute(
        f"UPDATE nudge_rules SET {cols} WHERE user_id = ? AND category = ?",
        (*sets.values(), user_id, category),
    )


def _recent_history(conn: sqlite3.Connection, user_id: str) -> List[str]:
    rows = conn.execute(
        "SELECT app FROM events WHERE user_id = ? AND event_type = 'Opened' "
        "ORDER BY timestamp DESC LIMIT ?",
        (user_id, RECENT_HISTORY),
    ).fetchall()
    return [a for (a,) in reversed(rows)]


def compute_next_action(
    conn: sqlite3.Connection,
    user_id: str,
    recommender,
    nudges,
    history: Optional[List[str]] = None,
) -> dict:
    ensure_defaults(conn, user_id, nudges)

    if history is None:
        history = _recent_history(conn, user_id)
    if not history:
        return {"action": None, "reason": "no_history"}

    preds = recommender.predict_next(history, k=1)
    if not preds:
        return {"action": None, "reason": "no_known_apps"}
    app = preds[0].item_id
    category = nudges.category_for(app)

    row = conn.execute(
        "SELECT category, enabled, message, image, notify, show_image, cooldown_seconds, last_fired "
        "FROM nudge_rules WHERE user_id = ? AND category = ?",
        (user_id, category),
    ).fetchone()
    base = {"predicted_app": app, "category": category}
    if row is None:
        return {"action": None, "reason": "no_rule", **base}
    rule = _row_to_rule(row)
    if not rule["enabled"]:
        return {"action": None, "reason": "disabled", **base}

    now = time.time()
    if now - rule["last_fired"] < rule["cooldown_seconds"]:
        return {"action": None, "reason": "cooldown", **base}

    message = rule["message"].strip() or random.choice(nudges.categories[category]["messages"])
    conn.execute(
        "UPDATE nudge_rules SET last_fired = ? WHERE user_id = ? AND category = ?",
        (now, user_id, category),
    )
    return {
        "action": {
            "category": category,
            "predicted_app": app,
            "message": message,
            "image": rule["image"] if rule["show_image"] else None,
            "notify": rule["notify"],
            "show_image": rule["show_image"],
        }
    }
