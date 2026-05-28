"""Re-evaluate a saved RecBole checkpoint with a custom topk list.

No retraining — loads the checkpoint, overrides topk, runs `trainer.evaluate`
on the held-out test split.

Usage:
    uv run python -m src.eval_topk saved/<checkpoint>.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# RecBole 1.2 still uses np.float (removed in numpy 1.24+). Patch before import.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from recbole.quick_start import load_data_and_model  # noqa: E402
from recbole.trainer import Trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument(
        "--topk", type=int, nargs="+", default=[1, 2, 3, 5, 10],
        help="k values to evaluate at",
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"not found: {args.checkpoint}")

    config, model, dataset, _train, _valid, test_data = load_data_and_model(
        model_file=str(args.checkpoint)
    )
    config["topk"] = args.topk
    config["metrics"] = ["Recall", "NDCG", "Hit"]
    config["valid_metric"] = f"NDCG@{max(args.topk)}"

    trainer = Trainer(config, model)
    result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    print(f"\n=== {args.checkpoint.name} ===")
    for k in args.topk:
        recall = result.get(f"recall@{k}")
        ndcg = result.get(f"ndcg@{k}")
        hit = result.get(f"hit@{k}")
        print(f"  @{k:<2}  recall={recall:.4f}  ndcg={ndcg:.4f}  hit={hit:.4f}")


if __name__ == "__main__":
    main()
