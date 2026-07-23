"""
MovieLens-1M loader with cold-start bins and leave-last-out split.

Dataset: https://files.grouplens.org/datasets/movielens/ml-1m.zip
Expected layout after extraction:
    data/raw/ml-1m/
        ratings.dat  (user_id::movie_id::rating::timestamp)
        movies.dat   (movie_id::title::genres)
        users.dat    (user_id::gender::age::occupation::zip)

Positive interactions are defined as rating >= 4 (standard choice in the
literature). Held-out split = each user's most-recent positive rating.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import ItemRecord, UserEvalSample, cold_start_bin


@dataclass
class MovieLensConfig:
    data_dir: str = "data/raw/ml-1m"
    positive_threshold: float = 4.0
    min_user_interactions: int = 5
    seed: int = 42


def _parse_dat(path: str, sep: str = "::", encoding: str = "latin-1") -> List[List[str]]:
    rows = []
    with open(path, "r", encoding=encoding) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(line.split(sep))
    return rows


class MovieLensLoader:
    """Load MovieLens-1M and expose cold-start-binned eval samples."""

    def __init__(self, config: Optional[MovieLensConfig] = None) -> None:
        self.config = config or MovieLensConfig()
        self.items: Dict[str, ItemRecord] = {}
        self.user_histories: Dict[str, List[Dict]] = {}

    def _paths(self) -> Tuple[str, str]:
        return (
            os.path.join(self.config.data_dir, "ratings.dat"),
            os.path.join(self.config.data_dir, "movies.dat"),
        )

    def is_downloaded(self) -> bool:
        r, m = self._paths()
        return os.path.isfile(r) and os.path.isfile(m)

    def load(self) -> None:
        """Load ratings + movies into memory."""
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"MovieLens-1M not found in {self.config.data_dir}. "
                f"Run scripts/download_movielens.sh first."
            )
        ratings_path, movies_path = self._paths()

        # --- movies ---
        for row in _parse_dat(movies_path):
            if len(row) < 3:
                continue
            mid, title, genres = row[0], row[1], row[2]
            self.items[mid] = ItemRecord(
                item_id=mid,
                title=title,
                category=genres,
                features={"genres": genres.split("|")},
            )

        # --- ratings ---
        interactions_by_user: Dict[str, List[Dict]] = defaultdict(list)
        for row in _parse_dat(ratings_path):
            if len(row) < 4:
                continue
            uid, mid, rating, ts = row[0], row[1], float(row[2]), int(row[3])
            interactions_by_user[uid].append(
                {"item_id": mid, "rating": rating, "timestamp": ts}
            )

        # sort each user's history by time
        for uid, inters in interactions_by_user.items():
            inters.sort(key=lambda x: x["timestamp"])
            self.user_histories[uid] = inters

    def build_eval_samples(
        self,
        cold_bins: Optional[List[int]] = None,
        max_users: Optional[int] = None,
    ) -> List[UserEvalSample]:
        """
        Build eval samples using leave-last-out:
          - The most recent *positive* interaction (rating >= threshold)
            becomes the held-out target.
          - Everything before that becomes the history.
          - history_length determines the cold-start bin.
        """
        if not self.user_histories:
            self.load()

        cold_bins_set = set(cold_bins) if cold_bins else None
        samples: List[UserEvalSample] = []
        for uid, inters in self.user_histories.items():
            # find the last positive interaction
            last_positive_idx = None
            for idx in range(len(inters) - 1, -1, -1):
                if inters[idx]["rating"] >= self.config.positive_threshold:
                    last_positive_idx = idx
                    break
            if last_positive_idx is None or last_positive_idx == 0:
                continue  # no positive or no history to use

            history = inters[:last_positive_idx]
            if len(history) < self.config.min_user_interactions:
                pass  # still keep — cold start is the point
            target_item = inters[last_positive_idx]["item_id"]

            bin_label = cold_start_bin(len(history))
            if cold_bins_set and len(history) not in cold_bins_set:
                # apply explicit bin filter if caller asked for it
                if bin_label != "50+" or 50 not in cold_bins_set:
                    continue

            sample = UserEvalSample(
                user_id=uid,
                history=history,
                relevant_items={target_item},
                cold_bin=bin_label,
                dataset="movielens-1m",
            )
            samples.append(sample)

            if max_users is not None and len(samples) >= max_users:
                break

        return samples

    def item_catalog(self) -> Dict[str, ItemRecord]:
        if not self.items:
            self.load()
        return self.items
