#!/usr/bin/env python3
"""
Statistical significance tests between CascadeRec and baselines.

Computes:
  - Paired t-test (parametric)
  - Wilcoxon signed-rank test (non-parametric)
  - Win/Tie/Loss counts
  - Effect size (Cohen's d)

Usage:
    python scripts/significance_test.py \
        experiments/movielens_always_small*.json \
        experiments/movielens_cascade_Rheuristic*.json
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def load_per_user(path):
    with open(path) as f:
        d = json.load(f)
    users = {}
    for u in d.get("per_user", []):
        users[u["user_id"]] = u
    return d.get("config", {}).get("method_name", "unknown"), users


def compare(name_a, users_a, name_b, users_b, metric="ndcg_at_10"):
    """Paired comparison on common users."""
    common = sorted(set(users_a.keys()) & set(users_b.keys()))
    if not common:
        print("No common users!")
        return
    va = np.array([users_a[u][metric] for u in common])
    vb = np.array([users_b[u][metric] for u in common])
    diff = vb - va

    # paired t-test
    t_stat, t_p = stats.ttest_rel(vb, va)
    # wilcoxon
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    # win/tie/loss
    wins = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    losses = int(np.sum(diff < 0))
    # Cohen's d
    sd = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / sd if sd > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Comparison: {name_a} vs {name_b}")
    print(f"Metric: {metric}")
    print(f"Common users: {len(common)}")
    print(f"{'='*60}")
    print(f"  {name_a} mean: {np.mean(va):.4f} ± {np.std(va):.4f}")
    print(f"  {name_b} mean: {np.mean(vb):.4f} ± {np.std(vb):.4f}")
    print(f"  Δ (B - A):     {np.mean(diff):.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"  Wilcoxon:      W={w_stat:.0f}, p={w_p:.4f}")
    print(f"  Cohen's d:     {cohens_d:.3f}")
    print(f"  W/T/L:         {wins}/{ties}/{losses}")
    sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
    print(f"  Significance:  {sig}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python significance_test.py baseline.json method.json [metric]")
        sys.exit(1)

    metric = sys.argv[3] if len(sys.argv) > 3 else "ndcg_at_10"
    name_a, users_a = load_per_user(sys.argv[1])
    name_b, users_b = load_per_user(sys.argv[2])
    compare(name_a, users_a, name_b, users_b, metric=metric)


if __name__ == "__main__":
    main()
