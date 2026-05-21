# Fitphone Backend

Backend for the Fontys Fitphone Streamlit app. Predicts the user's next likely app
from session history and returns a nudge (image + message) intended to interrupt
unhealthy smartphone habits.

## Architecture

```
Streamlit UI  ─HTTP─►  FastAPI  ─calls─►  RecBole model (SASRec / GRU4Rec)
                          │
                          ▼
                    SQLite (events, feedback)
```

- **Baselines** trained offline with RecBole on the LiveLab dataset.
- **FastAPI** loads a checkpoint once at startup and serves `/predict` + `/log_event`.
- **Streamlit** is pure UI: it calls the API and renders the nudge.
- **Personalization (phase 2)**: logged events accumulate in SQLite; a retrain script
  rewrites the checkpoint nightly.

## Phone-usage logging — status

Real on-device logging is **out of scope for the demo** (iOS has no public API;
Android needs a signed app with Usage Access permission). For the demo we:
1. Train on LiveLab.
2. Replay a LiveLab user's session through the Streamlit UI to mimic live logging.
3. Optionally accept a CSV upload (Android Digital Wellbeing export shape).

## Setup

Managed with [uv](https://docs.astral.sh/uv/). Python is pinned to **3.10/3.11** in
`pyproject.toml` because RecBole has install issues on 3.12+.

```bash
uv sync
```

Run anything with `uv run ...` (no manual venv activation needed).

## Data

We use **`appusage.sql`** from LiveLab (<https://yecl.org/livelab/traces.html>).
Schema: `id, uid, name, time (POSIX), duration` — one row per app foreground event.
That's exactly what a sequential next-app model needs, so it's the only file
required for the baseline. (`display.sql` from `power.tar.gz` is useful later for
defining screen-on sessions, but skip it for now.)

```bash
# 1. download apps.tar.gz from the LiveLab page into data/raw/
# 2. extract:
tar -xzf data/raw/apps.tar.gz -C data/raw/

# 3. convert appusage.sql -> RecBole atomic file:
uv run python -m src.convert_livelab \
    --sql data/raw/appusage.sql \
    --out data/recbole/fitphone
```

This writes `data/recbole/fitphone/fitphone.inter`. The converter parses
`INSERT INTO appusage VALUES (...)` statements directly, so it works whether
the dump is MySQL- or SQLite-flavored.

## Train

```bash
uv run python -m src.train --model SASRec     # transformer baseline
uv run python -m src.train --model GRU4Rec    # RNN baseline (the "LSTM-flavored" one)
```

Checkpoints land in `saved/`. Compare `Recall@10`, `NDCG@10`, `MRR@10` in the
training log.

## Run the API

```bash
uv run uvicorn src.api:app --reload --port 8000
```

The Streamlit frontend lives in a separate repo (owned by the team). It calls:

- `GET  /health`               — sanity check, reports whether a model is loaded
- `POST /predict`              — `{user_id, history: [app, ...], k}` → top-k + nudge
- `POST /log_event`            — `{user_id, app, timestamp?}` → appends to SQLite
- `GET  /recent/{user_id}`     — last N logged events for a user

## Repo layout

```
src/
  convert_livelab.py   LiveLab raw -> RecBole atomic files
  train.py             RecBole training entrypoint
  predict.py           Load checkpoint, predict top-k next apps
  api.py               FastAPI app
  nudges.py            Map predicted app -> nudge (image + message)
config/
  sasrec.yaml          RecBole config for SASRec
  gru4rec.yaml         RecBole config for GRU4Rec
assets/
  nudges.json          App category -> nudge mapping
  images/              Nudge images (placeholders for now)
```
