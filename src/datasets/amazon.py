"""
Amazon Reviews 2023 loader for CascadeRec.

Wraps the AmazonDataLoader already shipped in paper_a_github to keep a single
source of truth for Amazon ingestion, then emits `UserEvalSample` objects
with cold-start bins, matching MovieLens/LaMP-3 loaders.

We re-implement a slim version locally so that CascadeRec does not have to
depend on the paper_a_github package path — instead we copy only the parsing
logic we need and drop unused pieces (user behavior processor, seller constraints).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm import tqdm

from .base import ItemRecord, UserEvalSample, cold_start_bin

logger = logging.getLogger(__name__)

AMAZON_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"

AMAZON_URLS = {
    "Electronics": {
        "reviews": f"{AMAZON_BASE_URL}/review_categories/Electronics.jsonl.gz",
        "meta": f"{AMAZON_BASE_URL}/meta_categories/meta_Electronics.jsonl.gz",
    },
    "All_Beauty": {
        "reviews": f"{AMAZON_BASE_URL}/review_categories/All_Beauty.jsonl.gz",
        "meta": f"{AMAZON_BASE_URL}/meta_categories/meta_All_Beauty.jsonl.gz",
    },
}


@dataclass
class AmazonConfig:
    data_dir: str = "data/raw/amazon"
    category: str = "Electronics"
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    max_reviews: Optional[int] = None
    max_users: Optional[int] = None
    positive_threshold: float = 4.0
    seed: int = 42


class AmazonLoader:
    """Slim Amazon Reviews 2023 loader tuned for CascadeRec."""

    def __init__(self, config: Optional[AmazonConfig] = None) -> None:
        self.config = config or AmazonConfig()
        os.makedirs(self.config.data_dir, exist_ok=True)
        self.items: Dict[str, ItemRecord] = {}
        self.user_histories: Dict[str, List[Dict]] = {}

    def _paths(self):
        category = self.config.category
        return (
            os.path.join(self.config.data_dir, f"{category}_reviews.jsonl.gz"),
            os.path.join(self.config.data_dir, f"meta_{category}.jsonl.gz"),
        )

    def is_downloaded(self) -> bool:
        r, m = self._paths()
        return os.path.isfile(r) and os.path.isfile(m)

    def download(self) -> None:
        """Download raw files if missing."""
        import requests

        urls = AMAZON_URLS.get(self.config.category)
        if not urls:
            raise ValueError(f"Unknown Amazon category: {self.config.category}")
        reviews_path, meta_path = self._paths()
        for name, url, filepath in [
            ("reviews", urls["reviews"], reviews_path),
            ("meta", urls["meta"], meta_path),
        ]:
            if os.path.exists(filepath):
                logger.info(f"{name} already downloaded: {filepath}")
                continue
            logger.info(f"Downloading {name} from {url}")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(filepath, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=name
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    pbar.update(len(chunk))

    def load(self) -> None:
        if not self.is_downloaded():
            raise FileNotFoundError(
                f"Amazon {self.config.category} data not found in {self.config.data_dir}. "
                f"Call .download() first."
            )
        reviews_path, meta_path = self._paths()

        # --- metadata ---
        logger.info(f"Loading metadata from {meta_path}")
        with gzip.open(meta_path, "rt", encoding="utf-8") as fh:
            for line in tqdm(fh, desc="meta"):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = item.get("parent_asin") or item.get("asin")
                if not asin:
                    continue
                title = item.get("title", "")
                category = item.get("main_category", "")
                desc = item.get("description", "")
                if isinstance(desc, list):
                    desc = desc[0] if desc else ""
                self.items[asin] = ItemRecord(
                    item_id=asin,
                    title=title if isinstance(title, str) else "",
                    category=category if isinstance(category, str) else "",
                    description=desc if isinstance(desc, str) else "",
                    features={"price": item.get("price")},
                )

        # --- reviews ---
        logger.info(f"Loading reviews from {reviews_path}")
        raw_by_user: Dict[str, List[Dict]] = defaultdict(list)
        n_loaded = 0
        with gzip.open(reviews_path, "rt", encoding="utf-8") as fh:
            for line in tqdm(fh, desc="reviews"):
                if self.config.max_reviews and n_loaded >= self.config.max_reviews:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = r.get("user_id")
                iid = r.get("parent_asin") or r.get("asin")
                rating = r.get("rating")
                ts = r.get("timestamp")
                if not (uid and iid and rating is not None and ts is not None):
                    continue
                raw_by_user[uid].append(
                    {"item_id": iid, "rating": float(rating), "timestamp": int(ts)}
                )
                n_loaded += 1

        # --- item k-core filter (config.min_item_interactions) ---
        # Without this the catalog keeps millions of near-singleton items and
        # the iALS retriever's Recall@100 collapses to a few percent.
        item_counts: Dict[str, int] = defaultdict(int)
        for inters in raw_by_user.values():
            for x in inters:
                item_counts[x["item_id"]] += 1
        dense_items = {
            iid for iid, c in item_counts.items()
            if c >= self.config.min_item_interactions
        }
        logger.info(
            f"Item k-core (>= {self.config.min_item_interactions}): "
            f"{len(dense_items)}/{len(item_counts)} items kept"
        )

        # --- filter by min interactions ---
        filtered: Dict[str, List[Dict]] = {}
        for uid, inters in raw_by_user.items():
            inters = [x for x in inters if x["item_id"] in dense_items]
            if len(inters) < self.config.min_user_interactions:
                continue
            inters.sort(key=lambda x: x["timestamp"])
            filtered[uid] = inters

        # restrict catalog to surviving items (memory + retriever speed)
        self.items = {
            iid: rec for iid, rec in self.items.items() if iid in dense_items
        }

        # optional user subsample for wall-clock budget
        if self.config.max_users is not None and len(filtered) > self.config.max_users:
            rng = random.Random(self.config.seed)
            keep = rng.sample(list(filtered.keys()), self.config.max_users)
            filtered = {uid: filtered[uid] for uid in keep}

        self.user_histories = filtered
        logger.info(
            f"Amazon {self.config.category}: {len(self.items)} items, "
            f"{len(self.user_histories)} users after filtering"
        )

    def build_eval_samples(self) -> List[UserEvalSample]:
        """Leave-last-positive split, same protocol as MovieLensLoader."""
        if not self.user_histories:
            self.load()

        samples: List[UserEvalSample] = []
        for uid, inters in self.user_histories.items():
            last_positive_idx = None
            for idx in range(len(inters) - 1, -1, -1):
                if inters[idx]["rating"] >= self.config.positive_threshold:
                    last_positive_idx = idx
                    break
            if last_positive_idx is None or last_positive_idx == 0:
                continue
            history = inters[:last_positive_idx]
            target_item = inters[last_positive_idx]["item_id"]
            samples.append(
                UserEvalSample(
                    user_id=uid,
                    history=history,
                    relevant_items={target_item},
                    cold_bin=cold_start_bin(len(history)),
                    dataset=f"amazon-{self.config.category.lower()}",
                )
            )
        return samples

    def item_catalog(self) -> Dict[str, ItemRecord]:
        if not self.items:
            self.load()
        return self.items
