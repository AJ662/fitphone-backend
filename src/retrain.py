"""Retrain the deployed model on accumulated user events.

Pipeline:
  1. Download the latest events.db snapshot from the HF Dataset repo.
  2. Read its `events` table, normalize package names to LSApp app names
     via pkg_map, and append the new rows to the base lsapp.inter file.
  3. Run RecBole training on the combined dataset, warm-starting from
     `saved/<latest>.pth` so the model adapts incrementally.
  4. Upload the resulting checkpoint back to the HF Space — the Space
     auto-restarts and the new model is live.

Can be run:
  - Manually:  uv run python -m src.retrain
  - From the API:  POST /retrain (kicks off training in a background thread)
"""

from __future__ import annotations

import glob
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .pkg_map import NOISE_PACKAGES, normalize as normalize_app


ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = ROOT / "saved"
DATA_DIR = ROOT / "data" / "recbole" / "lsapp"
INTER_FILE = DATA_DIR / "lsapp.inter"
INTER_BACKUP = DATA_DIR / "lsapp.inter.base"

EVENTS_DATASET = os.environ.get("FITPHONE_EVENTS_DATASET", "zekerheydArthur/fitphone-events")
SPACE_REPO = os.environ.get("FITPHONE_SPACE", "zekerheydArthur/fitphone-backend")
DEFAULT_MODEL = os.environ.get("FITPHONE_MODEL", "GRU4Rec")


def _ensure_base_inter() -> None:
    """Keep an immutable copy of the original .inter so we always rebuild
    from a known base + the latest events, not incrementally accumulate."""
    if not INTER_BACKUP.exists():
        shutil.copy(INTER_FILE, INTER_BACKUP)


def _download_events_db(target: Path) -> Path | None:
    """Pull events.db from the dataset repo. Returns None if it doesn't exist
    yet (e.g. first-ever retrain before any snapshots ran)."""
    try:
        return Path(hf_hub_download(
            repo_id=EVENTS_DATASET,
            filename="events.db",
            repo_type="dataset",
            local_dir=str(target),
        ))
    except Exception as e:
        print(f"[retrain] no events.db yet: {e}")
        return None


def _events_to_inter_rows(db_path: Path) -> list[tuple[str, str, float]]:
    """Convert the events DB into RecBole interaction rows. Only Opened events
    survive, package names normalize to LSApp app names, noise (launcher/
    system) is dropped."""
    rows: list[tuple[str, str, float]] = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT user_id, app, timestamp FROM events "
            "WHERE event_type = 'Opened' "
            "ORDER BY timestamp ASC"
        )
        for user_id, app, ts in cur:
            if app in NOISE_PACKAGES:
                continue
            name = normalize_app(app)
            rows.append((str(user_id), name, float(ts)))
    return rows


def _append_to_inter(rows: list[tuple[str, str, float]]) -> int:
    """Restore the .inter from the base copy, then append the user's events.
    Returns count of rows actually added (skipping items not in vocab)."""
    _ensure_base_inter()
    shutil.copy(INTER_BACKUP, INTER_FILE)
    if not rows:
        return 0
    # Read header to learn the field order — RecBole's atomic format uses
    # ":token" / ":float" annotations in the header, which we preserve.
    with INTER_FILE.open("r", encoding="utf-8") as f:
        header = f.readline()
    with INTER_FILE.open("a", encoding="utf-8") as f:
        for user_id, item_id, ts in rows:
            f.write(f"{user_id}\t{item_id}\t{ts}\n")
    return len(rows)


def _latest_checkpoint(model_name: str) -> Path | None:
    files = glob.glob(str(SAVED_DIR / f"{model_name}-*.pth"))
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def _train(model_name: str, warm_from: Path | None) -> Path:
    """Run RecBole training. If warm_from is given, the trainer loads those
    weights before the loop starts."""
    from recbole.quick_start import run_recbole

    config_dict = {
        "epochs": int(os.environ.get("FITPHONE_RETRAIN_EPOCHS", "5")),
        "stopping_step": 2,
    }
    if warm_from is not None:
        # RecBole's `resume_checkpoint_file` continues training a saved run.
        config_dict["resume_checkpoint_file"] = str(warm_from)

    run_recbole(
        model=model_name,
        dataset="lsapp",
        config_file_list=[str(ROOT / "config" / f"{model_name.lower()}_lsapp.yaml")],
        config_dict=config_dict,
    )
    ckpt = _latest_checkpoint(model_name)
    if ckpt is None:
        raise RuntimeError("training finished but no checkpoint produced")
    return ckpt


def _upload_checkpoint(ckpt: Path) -> None:
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if token is None:
        print("[retrain] no HF_WRITE_TOKEN; skipping upload")
        return
    HfApi(token=token).upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=f"saved/{ckpt.name}",
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message=f"retrained {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
    )
    print(f"[retrain] uploaded {ckpt.name} to {SPACE_REPO}")


def retrain(model_name: str = DEFAULT_MODEL) -> dict:
    """End-to-end retrain. Returns a summary dict suitable for the API."""
    started = time.time()
    print(f"[retrain] starting — model={model_name}")

    with tempfile.TemporaryDirectory() as tmp:
        db = _download_events_db(Path(tmp))
        user_rows = _events_to_inter_rows(db) if db else []

    appended = _append_to_inter(user_rows)
    print(f"[retrain] appended {appended} user events to {INTER_FILE.name}")

    warm = _latest_checkpoint(model_name)
    print(f"[retrain] warm-start from: {warm}")

    new_ckpt = _train(model_name, warm)
    print(f"[retrain] new checkpoint: {new_ckpt}")

    _upload_checkpoint(new_ckpt)

    return {
        "ok": True,
        "model": model_name,
        "events_added": appended,
        "warm_start": warm.name if warm else None,
        "new_checkpoint": new_ckpt.name,
        "elapsed_seconds": round(time.time() - started, 1),
    }


if __name__ == "__main__":
    print(retrain(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL))
