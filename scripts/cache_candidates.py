#!/usr/bin/env python3
"""
Cache first-stage top-100 candidates per user × dataset.

Output: data/cache/{dataset}_{retriever}_top{k}.json
    {
      "meta": {...},
      "users": {
        "<user_id>": {
           "history_length": int,
           "cold_bin": str,
           "relevant": [...],
           "candidates": [...]
        }, ...
      }
    }

Usage:
    python scripts/cache_candidates.py --dataset lamp3 --retriever bpr --topk 100
    python scripts/cache_candidates.py --dataset movielens --retriever popularity --topk 100
    python scripts/cache_candidates.py --dataset amazon --retriever bpr --topk 100 --max-users 5000
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import BM25Retriever, BPRMFRetriever, PopularityRetriever
from src.datasets import (
    AmazonConfig,
    AmazonLoader,
    LaMP3Config,
    LaMP3Loader,
    MovieLensConfig,
    MovieLensLoader,
    build_cold_start_eval_set,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cache_candidates")


DATASET_CHOICES = {"movielens", "amazon", "lamp3"}
RETRIEVER_CHOICES = {"popularity", "bpr", "bm25"}


def build_retriever(name: str):
    if name == "popularity":
        return PopularityRetriever()
    if name == "bpr":
        return BPRMFRetriever(factors=64, iterations=50)
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unknown retriever: {name}")


def load_dataset(name: str, args):
    if name == "movielens":
        cfg = MovieLensConfig(data_dir=args.movielens_dir)
        loader = MovieLensLoader(cfg)
        samples = loader.build_eval_samples(max_users=args.max_users)
        return samples, loader.item_catalog()

    if name == "amazon":
        cfg = AmazonConfig(
            data_dir=args.amazon_dir,
            category=args.amazon_category,
            max_reviews=args.max_reviews,
            max_users=args.max_users,
            min_item_interactions=args.min_item_interactions,
            min_user_interactions=args.min_user_interactions,
        )
        loader = AmazonLoader(cfg)
        if not loader.is_downloaded():
            logger.info("Amazon data missing, downloading...")
            loader.download()
        samples = loader.build_eval_samples()
        return samples, loader.item_catalog()

    if name == "lamp3":
        cfg = LaMP3Config(
            data_dir=args.lamp_dir,
            split=args.lamp_split,
            few_shot_k=args.lamp_k,
            max_users=args.max_users,
        )
        loader = LaMP3Loader(cfg)
        samples = loader.load()
        return samples, loader.item_catalog()

    raise ValueError(f"Unknown dataset: {name}")


def _user_entry(s, cands: List[str]) -> Dict:
    return {
        "history_length": s.history_length,
        "cold_bin": s.cold_bin,
        "relevant": sorted(s.relevant_items),
        "candidates": cands,
        # real (truncated) history so run_experiment can build real prompts
        "history": [
            {"item_id": h["item_id"], "rating": h.get("rating")}
            for h in s.history
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--retriever", required=True, choices=RETRIEVER_CHOICES)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-reviews", type=int, default=None, help="Amazon only")
    parser.add_argument("--min-item-interactions", type=int, default=20, help="Amazon only")
    parser.add_argument("--candidate-protocol", choices=["retrieval", "sampled"],
                        default="retrieval",
                        help="'retrieval': top-k from full catalog; 'sampled': "
                             "target + N random negatives ranked by retriever "
                             "(standard for LLM reranking on large catalogs)")
    parser.add_argument("--n-negatives", type=int, default=19)
    parser.add_argument("--neg-seed", type=int, default=42)
    parser.add_argument("--min-user-interactions", type=int, default=5, help="Amazon only")
    parser.add_argument("--amazon-dir", type=str, default="data/raw/amazon")
    parser.add_argument("--amazon-category", type=str, default="Electronics")
    parser.add_argument("--movielens-dir", type=str, default="data/raw/ml-1m")
    parser.add_argument("--lamp-dir", type=str, default="data/raw/lamp")
    parser.add_argument("--lamp-split", type=str, default="dev")
    parser.add_argument("--lamp-k", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument(
        "--cold-start-ks",
        type=str,
        default="",
        help="Comma-separated list of k values to build artificial cold-start "
        "splits from (e.g. '1,3,5,10,20'). Empty = use original history.",
    )
    parser.add_argument(
        "--max-users-per-k",
        type=int,
        default=None,
        help="Cap users per cold-start k-bin (balanced evaluation).",
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    t0 = time.time()
    logger.info(f"Loading dataset={args.dataset} retriever={args.retriever} topk={args.topk}")
    warm_samples, catalog = load_dataset(args.dataset, args)
    logger.info(
        f"Loaded {len(warm_samples)} warm samples, catalog={len(catalog)} ({time.time()-t0:.1f}s)"
    )

    if not warm_samples:
        logger.error("No samples loaded — check dataset paths")
        sys.exit(1)

    # The retriever always trains on warm samples (full history) to avoid
    # leaking cold-start truncation into the first-stage model.
    t1 = time.time()
    retr = build_retriever(args.retriever)
    logger.info(f"Fitting retriever on warm samples: {args.retriever}")
    retr.fit(warm_samples, catalog)
    logger.info(f"Fit in {time.time()-t1:.1f}s")

    # Evaluation samples — either original (warm) or cold-start truncated
    if args.cold_start_ks:
        ks = [int(x) for x in args.cold_start_ks.split(",") if x.strip()]
        samples = build_cold_start_eval_set(
            warm_samples, ks, max_users_per_k=args.max_users_per_k
        )
        logger.info(
            f"Built cold-start eval set: ks={ks}, "
            f"users_per_k={args.max_users_per_k}, total={len(samples)}"
        )
    else:
        samples = warm_samples

    t2 = time.time()
    out_users: Dict[str, Dict] = {}
    if args.candidate_protocol == "sampled":
        # Sampled-negatives protocol (standard for LLM reranking on large
        # catalogs, cf. Hou et al. 2024): candidates = ground-truth target(s)
        # + N random negatives, ordered by retriever score. Guarantees the
        # target is in the candidate set; measures ranking ability.
        import random as _random

        logger.info(
            f"Caching sampled candidates (targets + {args.n_negatives} negatives)..."
        )
        rng = _random.Random(args.neg_seed)
        all_items = sorted(catalog.keys())
        for s in samples:
            exclude = set(s.relevant_items) | {h["item_id"] for h in s.history}
            negs: List[str] = []
            while len(negs) < args.n_negatives:
                iid = all_items[rng.randrange(len(all_items))]
                if iid not in exclude and iid not in negs:
                    negs.append(iid)
            pool = sorted(s.relevant_items) + negs
            cands = retr.rank_items(s, pool)
            out_users[s.user_id] = _user_entry(s, cands)
    else:
        logger.info("Caching top-k per user...")
        for s in samples:
            cands = retr.top_k(s, k=args.topk)
            out_users[s.user_id] = _user_entry(s, cands)
    logger.info(f"Ranked {len(out_users)} users in {time.time()-t2:.1f}s")

    # Minimal catalog for every item referenced by the cache (history +
    # candidates + relevant), so run_experiment can build REAL prompts —
    # history titles included — without reloading the raw dataset.
    referenced = set()
    for u in out_users.values():
        referenced.update(u["candidates"])
        referenced.update(u["relevant"])
        referenced.update(h["item_id"] for h in u["history"])
    out_catalog = {
        iid: {"title": catalog[iid].title, "category": catalog[iid].category}
        for iid in referenced
        if iid in catalog
    }
    logger.info(f"Embedded catalog: {len(out_catalog)}/{len(referenced)} referenced items")

    out = {
        "meta": {
            "dataset": args.dataset,
            "retriever": args.retriever,
            "topk": args.topk,
            "candidate_protocol": args.candidate_protocol,
            "n_negatives": args.n_negatives if args.candidate_protocol == "sampled" else None,
            "neg_seed": args.neg_seed if args.candidate_protocol == "sampled" else None,
            "n_users": len(out_users),
            "catalog_size": len(catalog),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "users": out_users,
        "catalog": out_catalog,
    }

    if args.candidate_protocol == "sampled":
        fname = f"{args.dataset}_{args.retriever}_sampled{args.n_negatives + 1}.json"
    else:
        fname = f"{args.dataset}_{args.retriever}_top{args.topk}.json"
    out_path = os.path.join(args.cache_dir, fname)
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    logger.info(f"Wrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")
    logger.info(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
