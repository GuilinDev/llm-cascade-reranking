#!/usr/bin/env python3
"""
CascadeRec experiment runner.

Usage:
    # B0: no-rerank baseline
    python scripts/run_experiment.py --method no_rerank --cache data/cache/movielens_bpr_top100.json --max-users 50

    # B1: always-small
    python scripts/run_experiment.py --method always_small --cache data/cache/movielens_bpr_top100.json --max-users 50

    # B2: always-large
    python scripts/run_experiment.py --method always_large --cache data/cache/movielens_bpr_top100.json --max-users 50

    # CascadeRec with heuristic router + large Tier-2
    python scripts/run_experiment.py --method cascade --router heuristic --tier2 large --cache data/cache/movielens_bpr_top100.json --max-users 50

    # CascadeRec with self-confidence router + multipath Tier-2
    python scripts/run_experiment.py --method cascade --router self_confidence --tier2 multipath --cache data/cache/movielens_bpr_top100.json --max-users 50
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cascade.cascade_rec import (
    AlwaysLargeBaseline,
    AlwaysSmallBaseline,
    CascadeConfig,
    CascadeRec,
    NoRerankBaseline,
)
from src.datasets.base import ItemRecord, UserEvalSample
from src.metrics.ranking import AggregateMetrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


def load_cache(cache_path: str, max_users: int = None, cold_bins: list = None):
    """Load cached candidates and rebuild UserEvalSample objects."""
    with open(cache_path) as f:
        cache = json.load(f)

    meta = cache["meta"]
    dataset = meta.get("dataset", "unknown")

    samples = []
    candidates_map = {}
    for uid, u in cache["users"].items():
        if cold_bins and u["cold_bin"] not in cold_bins:
            continue
        # Real truncated history when the cache provides it (caches built
        # after 2026-07-15); placeholder ids otherwise. Placeholder history
        # renders as meaningless "Item h_0" lines in LLM prompts — legacy
        # caches must be rebuilt before running LLM experiments.
        if u.get("history"):
            history = [
                {"item_id": h["item_id"], "rating": h.get("rating"), "timestamp": i}
                for i, h in enumerate(u["history"])
            ]
        else:
            history = [{"item_id": f"h_{i}", "rating": 3.0, "timestamp": i}
                       for i in range(u["history_length"])]
        sample = UserEvalSample(
            user_id=uid,
            history=history,
            relevant_items=set(u["relevant"]),
            cold_bin=u["cold_bin"],
            dataset=dataset,
            extra={
                "history_length": u["history_length"],
                "truncated_to": u["history_length"],
            },
        )
        samples.append(sample)
        candidates_map[uid] = u["candidates"]
        if max_users and len(samples) >= max_users:
            break

    return samples, candidates_map, meta


def build_catalog_from_cache(cache_path: str) -> dict:
    """Build a catalog from the cache. Caches built after 2026-07-15 embed
    real titles/categories; older caches degrade to Item-{id} placeholders,
    which produce meaningless LLM prompts — rebuild them instead."""
    with open(cache_path) as f:
        cache = json.load(f)

    embedded = cache.get("catalog")
    if embedded:
        return {
            iid: ItemRecord(item_id=iid, title=e.get("title") or f"Item-{iid}",
                            category=e.get("category") or "")
            for iid, e in embedded.items()
        }

    logger.warning(
        "Cache has no embedded catalog — prompts will use Item-{id} "
        "placeholders. Rebuild the cache with scripts/cache_candidates.py."
    )
    all_items = set()
    for u in cache["users"].values():
        all_items.update(u["candidates"])
        all_items.update(u["relevant"])

    return {
        iid: ItemRecord(item_id=iid, title=f"Item-{iid}", category="")
        for iid in all_items
    }


def load_movielens_catalog(data_dir: str = "data/raw/ml-1m") -> dict:
    """Load real MovieLens movie titles for LLM prompts."""
    movies_path = os.path.join(data_dir, "movies.dat")
    catalog = {}
    if os.path.isfile(movies_path):
        with open(movies_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 3:
                    mid, title, genres = parts[0], parts[1], parts[2]
                    catalog[mid] = ItemRecord(
                        item_id=mid,
                        title=title,
                        category=genres,
                        features={"genres": genres.split("|")},
                    )
    return catalog


def main():
    parser = argparse.ArgumentParser(description="CascadeRec experiment runner")
    parser.add_argument("--method", required=True,
                        choices=["no_rerank", "always_small", "always_large", "cascade"])
    parser.add_argument("--cache", required=True, help="Path to cached candidates JSON")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--cold-bins", type=str, default=None,
                        help="Comma-separated cold-start bins to include (e.g. '1,3,5')")
    parser.add_argument("--seed", type=int, default=42)

    # CascadeRec-specific
    parser.add_argument("--router", type=str, default="heuristic",
                        choices=["heuristic", "self_confidence", "logistic", "llm_coordinator"])
    parser.add_argument("--tier2", type=str, default="large",
                        choices=["large", "self_consistency", "multipath"])
    parser.add_argument("--tier1-model", type=str, default="mistral-small3.2:24b")
    parser.add_argument("--tier2-model", type=str, default="llama3.1:70b")
    parser.add_argument("--router-threshold", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--strict-output", action="store_true",
                        help="Append a list-only output constraint to prompts "
                             "(required for essay-prone models like qwen3.5)")
    parser.add_argument("--tier1-cache", type=str, default="auto",
                        help="'auto' (default): share a per-(dataset,model,seed) Tier-1 "
                             "cache under data/cache/tier1/; 'none' to disable; "
                             "or an explicit path")

    # Catalog
    parser.add_argument("--movielens-dir", type=str, default="data/raw/ml-1m")

    args = parser.parse_args()

    cold_bins = args.cold_bins.split(",") if args.cold_bins else None

    logger.info(f"Loading cache from {args.cache}")
    samples, candidates_map, cache_meta = load_cache(
        args.cache, max_users=args.max_users, cold_bins=cold_bins
    )
    logger.info(f"Loaded {len(samples)} samples")

    # Load real catalog if available
    dataset = cache_meta.get("dataset", "")
    if "movielens" in dataset and os.path.isdir(args.movielens_dir):
        catalog = load_movielens_catalog(args.movielens_dir)
        logger.info(f"Loaded MovieLens catalog: {len(catalog)} movies")
    else:
        catalog = build_catalog_from_cache(args.cache)
        logger.info(f"Using fallback catalog: {len(catalog)} items")

    # Resolve Tier-1 cache path. Always-large runs the big model as its
    # "tier 1", so its cache is keyed by the tier-2 model name.
    def resolve_tier1_cache(model: str):
        from src.cascade.tier1_cache import default_tier1_cache_path

        if args.tier1_cache == "none":
            return None
        if args.tier1_cache != "auto":
            return args.tier1_cache
        variant = "noconf" if args.method == "always_large" else "conf"
        if args.strict_output:
            variant += "_strict"
        return default_tier1_cache_path(
            dataset or "unknown", model, args.seed, args.max_candidates,
            variant=variant,
        )

    cache_model = args.tier2_model if args.method == "always_large" else args.tier1_model
    tier1_cache_path = resolve_tier1_cache(cache_model)
    if tier1_cache_path:
        logger.info(f"Tier-1 cache: {tier1_cache_path}")

    # Build method
    base_config = CascadeConfig(
        tier1_model=args.tier1_model,
        tier2_model=args.tier2_model,
        tier1_max_candidates=args.max_candidates,
        tier2_type=args.tier2,
        router_type=args.router,
        router_threshold=args.router_threshold,
        seed=args.seed,
        strict_output=args.strict_output,
        tier1_cache_path=tier1_cache_path,
    )

    if args.method == "no_rerank":
        method = NoRerankBaseline()
        method_name = "no_rerank"
    elif args.method == "always_small":
        method = AlwaysSmallBaseline(base_config)
        method_name = f"always_small_{args.tier1_model.replace(':', '_')}"
    elif args.method == "always_large":
        method = AlwaysLargeBaseline(base_config)
        method_name = f"always_large_{args.tier2_model.replace(':', '_')}"
    elif args.method == "cascade":
        method = CascadeRec(base_config)
        method_name = f"cascade_R{args.router}_T2{args.tier2}_{args.tier1_model.replace(':', '_')}"
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"Running {method_name} on {len(samples)} users...")
    t0 = time.time()
    results = method.run_experiment(samples, candidates_map, catalog)
    elapsed = time.time() - t0

    # Aggregate
    agg = AggregateMetrics.from_user_results(results)
    agg_dict = agg.as_dict()
    agg_dict["wall_clock_seconds"] = elapsed

    # Print summary
    print(f"\n{'='*60}")
    print(f"Method:        {method_name}")
    print(f"Dataset:       {dataset}")
    print(f"Users:         {agg.n_users}")
    print(f"NDCG@10:       {agg.ndcg_at_10:.4f}")
    print(f"NDCG@20:       {agg.ndcg_at_20:.4f}")
    print(f"Recall@20:     {agg.recall_at_20:.4f}")
    print(f"MRR:           {agg.mrr:.4f}")
    print(f"Hit@10:        {agg.hit_at_10:.4f}")
    print(f"Escalation:    {agg.escalation_rate:.1%}")
    print(f"T1 parse-fail: {agg.tier1_parse_failure_rate:.1%}")
    print(f"T2 parse-fail: {agg.tier2_parse_failure_rate:.1%}")
    print(f"Fallback rate: {agg.fallback_rate:.1%}")
    print(f"Avg tokens:    {agg.avg_tokens_per_user:.0f}")
    print(f"Avg latency:   {agg.avg_latency_ms_per_user:.0f} ms")
    print(f"Wall clock:    {elapsed:.1f}s")
    print(f"{'='*60}")

    if agg.per_bin:
        print(f"\n{'bin':>6} {'n':>5} {'NDCG@10':>10} {'Recall@20':>12} {'Escal%':>8}")
        for b in sorted(agg.per_bin.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            d = agg.per_bin[b]
            print(f"{b:>6} {d['n_users']:>5} {d['ndcg_at_10']:>10.4f} {d['recall_at_20']:>12.4f} {d['escalation_rate']:>7.1%}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"{dataset}_{method_name}_seed{args.seed}_{ts}.json")
    output = {
        "config": {
            "method": args.method,
            "method_name": method_name,
            "dataset": dataset,
            "cache_path": args.cache,
            "tier1_model": args.tier1_model,
            "tier2_model": args.tier2_model,
            "tier2_type": args.tier2,
            "router_type": args.router,
            "router_threshold": args.router_threshold,
            "max_candidates": args.max_candidates,
            "n_users": len(samples),
            "seed": args.seed,
            "tier1_cache_path": tier1_cache_path,
        },
        "aggregate_metrics": agg_dict,
        "per_user": [
            {
                "user_id": r.user_id,
                "cold_bin": r.cold_bin,
                "ndcg_at_10": r.ndcg_at_10,
                "recall_at_20": r.recall_at_20,
                "mrr": r.mrr,
                "escalated": r.escalated,
                "total_tokens": r.total_tokens,
                "total_latency_ms": r.total_latency_ms,
                "tier1_parse_ok": r.tier1_parse_ok,
                "tier2_parse_ok": r.tier2_parse_ok,
                "fallback": r.fallback,
            }
            for r in results  # save ALL users (needed for paired significance tests)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
