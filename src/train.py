"""Train a RecBole sequential model. Usage: python -m src.train --model SASRec"""

from __future__ import annotations

import argparse
from pathlib import Path

from recbole.quick_start import run_recbole


CONFIGS = {
    ("SASRec", "fitphone"): "config/sasrec.yaml",
    ("GRU4Rec", "fitphone"): "config/gru4rec.yaml",
    ("SASRec", "lsapp"): "config/sasrec_lsapp.yaml",
    ("GRU4Rec", "lsapp"): "config/gru4rec_lsapp.yaml",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["SASRec", "GRU4Rec"], default="SASRec")
    ap.add_argument("--dataset", choices=["fitphone", "lsapp"], default="fitphone")
    args = ap.parse_args()

    cfg = Path(CONFIGS[(args.model, args.dataset)])
    if not cfg.exists():
        raise SystemExit(f"missing config {cfg}")

    run_recbole(model=args.model, dataset=args.dataset, config_file_list=[str(cfg)])


if __name__ == "__main__":
    main()
