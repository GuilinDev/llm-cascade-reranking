#!/usr/bin/env python3
"""
Generate the IAAI-27 paper tables from experiments/iaai/*.json.

Outputs (ready to paste into paper_iaai/main.tex):
  Table MAIN    — per dataset: model x method aggregate (NDCG@10, MRR, Esc%, Tok)
  Table PERBIN  — per dataset: model x method NDCG@10 by cold-start bin,
                  with $^*$ where the 95% paired-bootstrap CI of the delta
                  vs B0 excludes zero
  Table DELTA   — per dataset: relative % change B1 vs B0 per bin
  Table COST    — per dataset: latency / tokens per user
  Table PARSE   — parse-failure & fallback rates per model (lessons section)

Usage:
    python scripts/fill_tables_iaai.py [--results-dir experiments/iaai]
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_iaai import B0_NAME, load_runs, paired_delta_ci  # noqa: E402

BINS = ["1", "3", "5", "10", "20"]

MODEL_LABELS = {
    "mistral-small3.2:24b": "Mistral-24B",
    "qwen3.5:27b": "Qwen-27B",
}

METHOD_LABELS = [
    # (method_name prefix, display label)
    ("no_rerank", "B0 No-Rerank"),
    ("always_small", "B1 Uniform"),
    ("cascade_Rheuristic_T2self_consistency", "CR($\\tau$=5)+SC"),
    ("cascade_Rheuristic_T2multipath", "CR($\\tau$=5)+MP"),
]

DATASET_LABELS = {"movielens": "MovieLens-1M", "amazon": "Amazon-Electronics"}


def method_label(method_name):
    for prefix, label in METHOD_LABELS:
        if method_name.startswith(prefix):
            return label
    return method_name


def model_of(run):
    cfg = run["config"]
    if cfg["method"] == "no_rerank":
        return None
    return MODEL_LABELS.get(cfg["tier1_model"], cfg["tier1_model"])


def star(deltas, rng):
    lo, hi = paired_delta_ci(np.asarray(deltas), rng)
    return "$^*$" if (hi < 0 or lo > 0) else ""


def rows_for(runs, ds):
    """Ordered (model_label, method_label, run) rows for one dataset."""
    out = []
    b0 = runs.get((ds, B0_NAME))
    if b0:
        out.append((None, "B0 No-Rerank", b0))
    keyed = {}
    for (d, m), run in runs.items():
        if d != ds or m == B0_NAME:
            continue
        keyed[(model_of(run), method_label(m))] = run
    for model in sorted({k[0] for k in keyed}):
        for _, label in METHOD_LABELS[1:]:
            if (model, label) in keyed:
                out.append((model, label, keyed[(model, label)]))
    return out, b0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/iaai")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        print(f"No results in {args.results_dir}")
        return
    rng = np.random.default_rng(42)

    for ds in sorted({k[0] for k in runs}):
        rows, b0 = rows_for(runs, ds)
        ds_label = DATASET_LABELS.get(ds, ds)
        n_users = rows[0][2]["config"]["n_users"] if rows else 0

        # ---------- MAIN ----------
        print(f"\n% ===== TABLE MAIN — {ds_label} ({n_users} users) =====")
        best = defaultdict(float)
        for model, label, run in rows:
            a = run["aggregate"]
            best[model] = max(best[model], a["ndcg_at_10"])
        for model, label, run in rows:
            a = run["aggregate"]
            nd = f"{a['ndcg_at_10']:.4f}"
            if a["ndcg_at_10"] == best[model]:
                nd = f"\\textbf{{{nd}}}"
            print(f"{model or '--'} & {label} & {nd} & {a['mrr']:.4f} & "
                  f"{a['escalation_rate']*100:.0f}\\% & "
                  f"{a['avg_tokens_per_user']:.0f} \\\\")

        # ---------- PERBIN ----------
        print(f"\n% ===== TABLE PERBIN — {ds_label} (NDCG@10 by k; * = 95% "
              f"paired-bootstrap CI of delta vs B0 excludes 0) =====")
        for model, label, run in rows:
            cells = []
            for b in BINS:
                pb = run["aggregate"].get("per_bin", {}).get(b)
                if pb is None:
                    cells.append("--")
                    continue
                cell = f"{pb['ndcg_at_10']:.3f}".lstrip("0") or "0"
                if b0 is not None and run is not b0:
                    common = [u for u, r in run["users"].items()
                              if r["cold_bin"] == b and u in b0["users"]]
                    if common:
                        d = [run["users"][u]["ndcg_at_10"] - b0["users"][u]["ndcg_at_10"]
                             for u in common]
                        cell += star(d, rng)
                cells.append(cell)
            print(f"{model or '--'} & {label} & " + " & ".join(cells) + " \\\\")

        # ---------- DELTA (B1 vs B0 relative %) ----------
        if b0 is not None:
            print(f"\n% ===== TABLE DELTA — {ds_label} (rel. % change B1 vs B0 "
                  f"per bin) =====")
            b1s = [(m, r) for m, l, r in rows if l == "B1 Uniform"]
            for b in BINS:
                cells = []
                for model, run in b1s:
                    pb = run["aggregate"]["per_bin"].get(b)
                    pb0 = b0["aggregate"]["per_bin"].get(b)
                    if not pb or not pb0 or not pb0["ndcg_at_10"]:
                        cells.append("--")
                        continue
                    rel = 100 * (pb["ndcg_at_10"] - pb0["ndcg_at_10"]) / pb0["ndcg_at_10"]
                    cells.append(f"${rel:+.1f}$")
                print(f"{b} & " + " & ".join(cells) + " \\\\")

        # ---------- COST ----------
        print(f"\n% ===== TABLE COST — {ds_label} (latency / tokens per user) =====")
        for model, label, run in rows:
            a = run["aggregate"]
            lat = a["avg_latency_ms_per_user"] / 1000
            print(f"{model or '--'} & {label} & {lat:.1f}s & "
                  f"{a['avg_tokens_per_user']:.0f} \\\\")

        # ---------- PARSE ----------
        print(f"\n% ===== TABLE PARSE — {ds_label} (serving-robustness stats) =====")
        for model, label, run in rows:
            a = run["aggregate"]
            print(f"{model or '--'} & {label} & "
                  f"{a.get('tier1_parse_failure_rate',0)*100:.1f}\\% & "
                  f"{a.get('tier2_parse_failure_rate',0)*100:.1f}\\% & "
                  f"{a.get('fallback_rate',0)*100:.1f}\\% \\\\")


if __name__ == "__main__":
    main()
