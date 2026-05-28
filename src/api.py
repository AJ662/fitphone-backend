"""FastAPI backend. Loads a RecBole checkpoint once and serves predictions + nudges."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .habits import get_habits, init_habits_db, log_habit
from .nudges import NudgeBook
from .pkg_map import NOISE_PACKAGES, normalize as normalize_app
from .predict import Recommender
from .rules import compute_next_action, get_rules, init_rules_db, update_rule
from .stats import compute_stats


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "events.db"
NUDGES_PATH = ROOT / "assets" / "nudges.json"
SAVED_DIR = ROOT / "saved"
ASSETS_DIR = ROOT / "assets"


class PredictRequest(BaseModel):
    user_id: str
    history: List[str]
    k: int = 5


class PredictResponse(BaseModel):
    top: list
    nudge: dict


class LogEvent(BaseModel):
    user_id: str
    app: str
    event_type: str = "Opened"
    timestamp: Optional[float] = None


class HabitLog(BaseModel):
    habit_key: str
    timestamp: Optional[float] = None


class RuleUpdate(BaseModel):
    enabled: Optional[bool] = None
    message: Optional[str] = None
    notify: Optional[bool] = None
    show_image: Optional[bool] = None
    cooldown_seconds: Optional[int] = None


class NextActionRequest(BaseModel):
    user_id: str
    history: Optional[List[str]] = None


# A new session starts when a user is idle longer than this (seconds).
SESSION_GAP_SECONDS = 5 * 60


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "user_id TEXT NOT NULL,"
            "app TEXT NOT NULL,"
            "timestamp REAL NOT NULL"
            ")"
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        if "session_id" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN session_id INTEGER")
        if "event_type" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN event_type TEXT NOT NULL DEFAULT 'Opened'")
        init_habits_db(conn)
        init_rules_db(conn)


def session_id_for(conn: sqlite3.Connection, user_id: str, ts: float) -> int:
    """Reuse the user's last session if they were recently active, else start a new one."""
    row = conn.execute(
        "SELECT session_id, timestamp FROM events WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return 1
    last_session, last_ts = row
    if ts - last_ts <= SESSION_GAP_SECONDS:
        return last_session
    return last_session + 1


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    state["nudges"] = NudgeBook(NUDGES_PATH)
    model_name = os.environ.get("FITPHONE_MODEL")  # optional: "SASRec" or "GRU4Rec"
    try:
        state["recommender"] = Recommender.from_latest(str(SAVED_DIR), model_name)
    except FileNotFoundError:
        state["recommender"] = None  # API still runs; /predict will 503
    yield


app = FastAPI(title="Fitphone Backend", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state.get("recommender") is not None}


@app.get("/apps")
def apps() -> dict:
    rec = state.get("recommender")
    if rec is None:
        raise HTTPException(status_code=503, detail="no model loaded; train first")
    field = rec.config["ITEM_ID_FIELD"]
    names = [t for t in rec.dataset.field2token_id[field] if t != "[PAD]"]
    return {"apps": sorted(names)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    rec = state.get("recommender")
    if rec is None:
        raise HTTPException(status_code=503, detail="no model loaded; train first")
    history = [normalize_app(a) for a in req.history if a not in NOISE_PACKAGES]
    preds = rec.predict_next(history, k=req.k)
    if not preds:
        raise HTTPException(status_code=400, detail="history contains no known apps")
    nudge = state["nudges"].nudge_for(preds[0].item_id)
    return PredictResponse(
        top=[{"app": p.item_id, "score": p.score} for p in preds],
        nudge={"message": nudge.message, "image_path": nudge.image_path, "category": nudge.category},
    )


@app.post("/log_event")
def log_event(ev: LogEvent) -> dict:
    ts = ev.timestamp if ev.timestamp is not None else time.time()
    with sqlite3.connect(DB_PATH) as conn:
        session_id = session_id_for(conn, ev.user_id, ts)
        conn.execute(
            "INSERT INTO events (user_id, session_id, app, event_type, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (ev.user_id, session_id, ev.app, ev.event_type, ts),
        )
    return {"ok": True, "session_id": session_id}


@app.post("/log_events")
def log_events(events: list[LogEvent]) -> dict:
    if not events:
        return {"ok": True, "inserted": 0}
    with sqlite3.connect(DB_PATH) as conn:
        rows = []
        for ev in events:
            ts = ev.timestamp if ev.timestamp is not None else time.time()
            session_id = session_id_for(conn, ev.user_id, ts)
            rows.append((ev.user_id, session_id, ev.app, ev.event_type, ts))
        conn.executemany(
            "INSERT INTO events (user_id, session_id, app, event_type, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    return {"ok": True, "inserted": len(rows)}


@app.get("/recent/{user_id}")
def recent(user_id: str, limit: int = 20) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT app, timestamp FROM events WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return {"events": [{"app": a, "timestamp": t} for a, t in rows]}


@app.get("/stats/{user_id}")
def stats(user_id: str) -> dict:
    nudges = state.get("nudges")
    social = set(nudges.categories.get("social", {}).get("apps", [])) if nudges else set()
    with sqlite3.connect(DB_PATH) as conn:
        return compute_stats(conn, user_id, social)


@app.get("/habits/{user_id}")
def habits(user_id: str) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        return get_habits(conn, user_id)


@app.post("/habits/{user_id}/log")
def habits_log(user_id: str, body: HabitLog) -> dict:
    ts = body.timestamp if body.timestamp is not None else time.time()
    with sqlite3.connect(DB_PATH) as conn:
        log_habit(conn, user_id, body.habit_key, ts)
    return {"ok": True}


@app.get("/rules/{user_id}")
def rules(user_id: str) -> dict:
    nudges = state.get("nudges")
    with sqlite3.connect(DB_PATH) as conn:
        return {"rules": get_rules(conn, user_id, nudges)}


@app.put("/rules/{user_id}/{category}")
def rule_update(user_id: str, category: str, body: RuleUpdate) -> dict:
    nudges = state.get("nudges")
    with sqlite3.connect(DB_PATH) as conn:
        get_rules(conn, user_id, nudges)  # ensure defaults exist
        update_rule(conn, user_id, category, body.model_dump(exclude_none=True))
    return {"ok": True}


@app.post("/next_action")
def next_action(req: NextActionRequest) -> dict:
    rec = state.get("recommender")
    if rec is None:
        raise HTTPException(status_code=503, detail="no model loaded; train first")
    nudges = state["nudges"]
    with sqlite3.connect(DB_PATH) as conn:
        return compute_next_action(conn, req.user_id, rec, nudges, req.history)


@app.get("/export_tsv", response_class=PlainTextResponse)
def export_tsv() -> str:
    """Dump the event log in LSApp TSV column order so convert_lsapp.py runs unchanged."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT user_id, session_id, timestamp, app, event_type "
            "FROM events ORDER BY user_id, timestamp"
        ).fetchall()
    lines = ["user_id\tsession_id\ttimestamp\tapp_name\tevent_type"]
    for user_id, session_id, ts, app, event_type in rows:
        when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"{user_id}\t{session_id}\t{when}\t{app}\t{event_type}")
    return "\n".join(lines) + "\n"
