"""Map a predicted app (or its category) to a nudge."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Nudge:
    message: str
    image_path: str
    category: str


class NudgeBook:
    def __init__(self, path: Path):
        data = json.loads(path.read_text())
        self.categories: Dict[str, dict] = data["categories"]
        self.app_to_category: Dict[str, str] = {
            app.lower(): cat for cat, body in self.categories.items() for app in body.get("apps", [])
        }
        self.default_category = data.get("default", "default")

    def category_for(self, app: str) -> str:
        return self.app_to_category.get(app.lower(), self.default_category)

    def nudge_for(self, app: str) -> Nudge:
        category = self.category_for(app)
        body = self.categories[category]
        return Nudge(
            message=random.choice(body["messages"]),
            image_path=body["image"],
            category=category,
        )
