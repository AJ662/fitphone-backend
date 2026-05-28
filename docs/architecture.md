# Architecture

```
┌─────────────┐    HTTP     ┌──────────────┐    in-proc    ┌─────────────┐
│  Streamlit  │ ──────────► │  FastAPI     │ ────────────► │  RecBole    │
│  (frontend) │ ◄────────── │  (backend)   │ ◄──────────── │  GRU4Rec    │
└─────────────┘   JSON      └──────┬───────┘               └─────────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  SQLite      │  ← logged user events
                            │  events.db   │
                            └──────────────┘
```

## Components

### Streamlit frontend
Lives in a separate repo (owned by another team member). Pure UI: takes a user
history, calls the API, renders the nudge.

### FastAPI backend (`src/api.py`)
Loads one RecBole checkpoint at startup and serves four endpoints:

| Endpoint              | Method | Purpose                                   |
|-----------------------|--------|-------------------------------------------|
| `/health`             | GET    | Sanity check, reports model_loaded status |
| `/predict`            | POST   | Top-k next apps + nudge                   |
| `/log_event`          | POST   | Append an event to SQLite                 |
| `/recent/{user_id}`   | GET    | Read back the last N events for a user    |

Auto-generated Swagger UI at `/docs`.

### Model (`src/predict.py`)
Wraps a RecBole checkpoint for inference. The recommender is **sequence-only**:
it predicts the next app from a sliding window of the user's recent history
(max 50). It does *not* use a per-user embedding — there is no user-id
cold-start problem.

### Nudge logic (`src/nudges.py`, `assets/nudges.json`)
Maps the top-1 predicted app to a nudge category (social / video / messaging /
shopping / games / default), then picks a random message from that category.
Image paths point to `assets/images/`.

### Event log (`events.db`, SQLite)
Schema:
```sql
events(id, user_id TEXT, app TEXT, timestamp REAL)
```
Appended on every `/log_event`. Used for offline retraining in phase 2.

## Why this shape and not "Streamlit calls RecBole directly"

Streamlit reruns its entire script on every interaction. A 100 MB Torch model
would reload every click. FastAPI keeps the model warm in memory; Streamlit
becomes a thin client.

## Phone-usage logging — out of scope

Real on-device app-usage capture is platform-restricted:
- **iOS:** Screen Time API is not exposed to third-party apps.
- **Android:** Requires a signed app with the `PACKAGE_USAGE_STATS`
  permission and a manual grant in Settings → "Usage access".

Neither is realistic for a short school-project deadline. The demo simulates
on-device logging by accepting hand-typed or CSV-uploaded history through the
Streamlit UI.

## Train → serve pipeline

```
data/raw/<lsapp.tsv | appusage.sql>
        │
        ▼  src/convert_lsapp.py  /  src/convert_livelab.py
data/recbole/<lsapp | fitphone>/<name>.inter
        │
        ▼  src/train.py  (reads config/<model>_<dataset>.yaml)
saved/<Model>-<timestamp>.pth
        │
        ▼  src/api.py  (auto-loads most recent checkpoint at startup)
        │
        ▼  HTTP
clients (Streamlit, curl, etc.)
```

Force a specific model at startup:
```bash
FITPHONE_MODEL=SASRec uv run uvicorn src.api:app --port 8000
```
