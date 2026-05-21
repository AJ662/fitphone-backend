"""FastAPI backend. Loads a RecBole checkpoint once and serves predictions + nudges."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .nudges import NudgeBook
from .predict import Recommender


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "events.db"
NUDGES_PATH = ROOT / "assets" / "nudges.json"
SAVED_DIR = ROOT / "saved"


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
    timestamp: Optional[float] = None


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": state.get("recommender") is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    rec = state.get("recommender")
    if rec is None:
        raise HTTPException(status_code=503, detail="no model loaded; train first")
    preds = rec.predict_next(req.history, k=req.k)
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
        conn.execute(
            "INSERT INTO events (user_id, app, timestamp) VALUES (?, ?, ?)",
            (ev.user_id, ev.app, ts),
        )
    return {"ok": True}


@app.get("/recent/{user_id}")
def recent(user_id: str, limit: int = 20) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT app, timestamp FROM events WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return {"events": [{"app": a, "timestamp": t} for a, t in rows]}
