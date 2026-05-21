"""Load a trained RecBole checkpoint and serve top-k next-app predictions."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

import torch
from recbole.data.interaction import Interaction
from recbole.quick_start import load_data_and_model


@dataclass
class Prediction:
    item_id: str
    score: float


class Recommender:
    """Wraps a RecBole checkpoint for inference.

    RecBole stores user/item id <-> internal-index mappings on the dataset
    object. Sequential models score the *next* item given a history sequence,
    so we feed the recent app history through the model and read off the top-k.
    """

    def __init__(self, checkpoint_path: str):
        self.config, self.model, self.dataset, _, _, _ = load_data_and_model(
            model_file=checkpoint_path
        )
        self.model.eval()
        self.device = self.config["device"]

    @classmethod
    def from_latest(cls, saved_dir: str = "saved", model_name: str | None = None) -> "Recommender":
        pattern = f"{model_name}-*.pth" if model_name else "*.pth"
        files = glob.glob(os.path.join(saved_dir, pattern))
        if not files:
            raise FileNotFoundError(f"no checkpoint under {saved_dir} matching {pattern}")
        latest = max(files, key=os.path.getmtime)
        return cls(latest)

    def predict_next(self, history: List[str], k: int = 5) -> List[Prediction]:
        item_field = self.config["ITEM_ID_FIELD"]
        item_list_field = item_field + "_list"
        item_length_field = "item_length"

        token2id = self.dataset.field2token_id[item_field]
        id2token = {v: k for k, v in token2id.items()}

        max_len = self.config["MAX_ITEM_LIST_LENGTH"]
        history = history[-max_len:]
        ids = [token2id[h] for h in history if h in token2id]
        if not ids:
            return []

        seq = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        seq[0, : len(ids)] = torch.tensor(ids, device=self.device)
        length = torch.tensor([len(ids)], device=self.device)

        interaction = Interaction({
            item_list_field: seq,
            item_length_field: length,
        }).to(self.device)

        with torch.no_grad():
            scores = self.model.full_sort_predict(interaction).squeeze(0)
        scores[0] = -float("inf")  # mask padding id

        top_scores, top_indices = torch.topk(scores, k=k)
        out: List[Prediction] = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            token = id2token.get(idx)
            if token is None:
                continue
            out.append(Prediction(item_id=str(token), score=float(score)))
        return out
