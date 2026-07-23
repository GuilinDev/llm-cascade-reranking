"""
LaMP-3 loader for CascadeRec.

LaMP-3 is personalized product rating (classification 1-5). We wrap it in
the same `UserEvalSample` interface as MovieLens/Amazon so CascadeRec
treats all three datasets uniformly.

For LaMP-3 specifically:
  - `history` = the user's profile entries (previous review-rating pairs)
  - `candidate_items` = rating buckets {1,2,3,4,5} (used as item-ids)
  - `relevant_items` = the ground-truth rating as a singleton set
  - `input_text` = the new product description to rate

This means "reranking" on LaMP-3 reduces to ranking the 5 rating buckets,
which degenerates to classification but keeps the CascadeRec evaluation
pipeline uniform.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import ItemRecord, UserEvalSample, cold_start_bin


@dataclass
class LaMP3Config:
    data_dir: str = ""  # path to directory containing LaMP_3/
    split: str = "dev"  # dev or train
    few_shot_k: int = 5  # truncate profile to last-k entries (cold-start sim)
    max_users: Optional[int] = None
    seed: int = 42


class LaMP3Loader:
    """LaMP-3 product rating loader."""

    def __init__(self, config: Optional[LaMP3Config] = None) -> None:
        self.config = config or LaMP3Config()
        self.items: Dict[str, ItemRecord] = {
            str(r): ItemRecord(
                item_id=str(r),
                title=f"rating-{r}",
                category="rating",
            )
            for r in range(1, 6)
        }

    def _split_dir(self) -> str:
        return os.path.join(self.config.data_dir, "LaMP_3", self.config.split)

    def _split_files(self):
        d = self._split_dir()
        split = self.config.split
        # LaMP repo uses `{split}_questions.json` and `{split}_outputs.json`
        return (
            os.path.join(d, f"{split}_questions.json"),
            os.path.join(d, f"{split}_outputs.json"),
        )

    def is_downloaded(self) -> bool:
        qpath, opath = self._split_files()
        return os.path.isfile(qpath) and os.path.isfile(opath)

    def load(self) -> List[UserEvalSample]:
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"LaMP-3 split not found in {self._split_dir()}. "
                f"Run mppr-personalization/scripts/download_data.sh first."
            )
        qpath, opath = self._split_files()
        with open(qpath, "r") as fh:
            questions = json.load(fh)
        with open(opath, "r") as fh:
            outputs_raw = json.load(fh)
        outputs = {o["id"]: o["output"] for o in outputs_raw.get("golds", outputs_raw)}

        samples: List[UserEvalSample] = []
        for q in questions:
            uid = str(q["id"])
            profile = q.get("profile", [])[-self.config.few_shot_k :]
            input_text = q.get("input", "")
            gold = outputs.get(q["id"]) or outputs.get(uid)
            if gold is None:
                continue
            history = [
                {
                    "item_id": str(p.get("score", "")),
                    "rating": float(p.get("score", 3)),
                    "text": p.get("text", ""),
                    "timestamp": 0,
                }
                for p in profile
            ]
            samples.append(
                UserEvalSample(
                    user_id=uid,
                    history=history,
                    candidate_items=[str(i) for i in range(1, 6)],
                    relevant_items={str(gold)},
                    cold_bin=cold_start_bin(len(history)),
                    dataset="lamp-3",
                    input_text=input_text,
                )
            )
            if self.config.max_users is not None and len(samples) >= self.config.max_users:
                break
        return samples

    def item_catalog(self) -> Dict[str, ItemRecord]:
        return self.items
