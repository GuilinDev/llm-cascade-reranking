#!/usr/bin/env python3
"""
Statistical analysis for the IAAI offline matrix (RecSys R1-W4 / R2 / R3:
"no statistical support").

For each (dataset, model): pairs users across methods by user_id, then
reports, per bin and aggregate:
  - mean NDCG@10 with 95% paired-bootstrap CI on the delta vs B0
  - Wilcoxon signed-rank p-value vs B0
  - parse-failure / fallback rates

Usage:
    python scripts/analyze_iaai.py [--results-dir experiments/iaai] [--latex]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy always present per requirements, but be safe
    wilcoxon = None

B0_NAME = "no_rerank"
N_BOOT = 10_000
SEED = 42


def load_runs(results_dir):
    """{(dataset, method_name): {user_id: per-user dict}} — latest file wins."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        d = json.load(open(path))
        cfg = d["config"]
        key = (cfg["dataset"], cfg["method_name"])
        runs[key] = {
            "users": {u["user_id"]: u for u in d["per_user"]},
            "aggregate": d["aggregate_metrics"],
            "config": cfg,
            "path": path,
        }
    return runs


def paired_delta_ci(deltas, rng):
    idx = rng.integers(0, len(deltas), size=(N_BOOT, len(deltas)))
    boots = deltas[idx].mean(axis=1)
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def analyze(runs, latex=False):
    datasets = sorted({k[0] for k in runs})
    rng = np.random.default_rng(SEED)
    for ds in datasets:
        b0 = runs.get((ds, B0_NAME))
        if b0 is None:
            print(f"[{ds}] no B0 run — skipping")
            continue
        methods = sorted(m for (d, m) in runs if d == ds and m != B0_NAME)
        print(f"\n{'='*100}\nDATASET: {ds}   (B0 = {B0_NAME}, paired on user_id, "
              f"{N_BOOT} bootstrap resamples)\n{'='*100}")
        for m in methods:
            run = runs[(ds, m)]
            common = sorted(set(run["users"]) & set(b0["users"]))
            if not common:
                print(f"{m}: no paired users")
                continue
            x = np.array([run["users"][u]["ndcg_at_10"] for u in common])
            y = np.array([b0["users"][u]["ndcg_at_10"] for u in common])
            deltas = x - y
            lo, hi = paired_delta_ci(deltas, rng)
            if wilcoxon is not None and np.any(deltas != 0):
                p = wilcoxon(x, y, zero_method="zsplit").pvalue
            else:
                p = float("nan")
            sig = "*" if hi < 0 or lo > 0 else " "
            print(f"\n{m}  (n={len(common)})")
            print(f"  aggregate: NDCG@10={x.mean():.4f} vs B0={y.mean():.4f}  "
                  f"Δ={deltas.mean():+.4f} CI95=[{lo:+.4f},{hi:+.4f}]{sig} "
                  f"Wilcoxon p={p:.4f}")
            # per bin
            bins = defaultdict(list)
            for u in common:
                bins[run["users"][u]["cold_bin"]].append(u)
            rows = []
            for b in sorted(bins, key=lambda z: int(z) if z.isdigit() else 999):
                us = bins[b]
                xb = np.array([run["users"][u]["ndcg_at_10"] for u in us])
                yb = np.array([b0["users"][u]["ndcg_at_10"] for u in us])
                db = xb - yb
                blo, bhi = paired_delta_ci(db, rng)
                bsig = "*" if bhi < 0 or blo > 0 else " "
                rel = 100 * db.mean() / yb.mean() if yb.mean() else float("nan")
                rows.append((b, len(us), xb.mean(), yb.mean(), db.mean(), blo, bhi, rel, bsig))
                print(f"  k={b:>3} n={len(us):>4}  {xb.mean():.4f} vs {yb.mean():.4f}  "
                      f"Δ={db.mean():+.4f} [{blo:+.4f},{bhi:+.4f}]{bsig}  rel={rel:+.1f}%")
            if latex:
                cells = " & ".join(
                    f"{r[2]:.3f}{'$^*$' if r[8]=='*' else ''}" for r in rows
                )
                print(f"  LaTeX: {m} & {cells} \\\\")
            agg = run["aggregate"]
            print(f"  parse-fail: T1={agg.get('tier1_parse_failure_rate',0):.1%} "
                  f"T2={agg.get('tier2_parse_failure_rate',0):.1%} "
                  f"fallback={agg.get('fallback_rate',0):.1%} "
                  f"esc={agg.get('escalation_rate',0):.0%} "
                  f"tok/user={agg.get('avg_tokens_per_user',0):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/iaai")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        print(f"No results in {args.results_dir}")
        return
    analyze(runs, latex=args.latex)


if __name__ == "__main__":
    main()
