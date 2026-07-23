#!/usr/bin/env python3
"""
Generate publication figures for CascadeRec.

Reads experiments/*.json results files and produces:
  - figures/pareto_curve.pdf    (NDCG@10 vs tokens/user)
  - figures/perbin_heatmap.pdf  (per cold-start bin breakdown)
  - figures/router_comparison.pdf
  - figures/tier2_comparison.pdf

Usage:
    python scripts/generate_figures.py --results-dir experiments --output-dir figures
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> dict:
    """Load all experiment results into a dict keyed by method_name."""
    results = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        name = data.get("config", {}).get("method_name", os.path.basename(path))
        results[name] = data
    return results


def plot_pareto(results: dict, output_path: str):
    """NDCG@10 vs average tokens per user — Pareto frontier."""
    fig, ax = plt.subplots(figsize=(6, 4))

    methods = []
    for name, data in results.items():
        agg = data.get("aggregate_metrics", {})
        ndcg = agg.get("ndcg_at_10", 0)
        tokens = agg.get("avg_tokens_per_user", 0)
        methods.append((name, tokens, ndcg))

    # Group by type for coloring
    for name, tokens, ndcg in methods:
        if "no_rerank" in name:
            marker, color, label = "s", "gray", "B0: No Rerank"
        elif "always_small" in name:
            marker, color, label = "^", "blue", "B1: Always Small"
        elif "always_large" in name:
            marker, color, label = "v", "red", "B2: Always Large"
        elif "cascade" in name:
            marker, color, label = "o", "green", name.replace("cascade_", "CR: ")
        else:
            marker, color, label = "x", "black", name

        ax.scatter(tokens, ndcg, marker=marker, color=color, s=100, zorder=5)
        ax.annotate(label, (tokens, ndcg), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Average Tokens per User", fontsize=11)
    ax.set_ylabel("NDCG@10", fontsize=11)
    ax.set_title("Cost-Quality Pareto Frontier", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_perbin(results: dict, output_path: str):
    """Per cold-start bin NDCG@10 grouped bar chart."""
    bins = ["1", "3", "5", "10", "20"]

    # Select key methods to compare
    key_methods = {}
    for name, data in results.items():
        if any(k in name for k in ["no_rerank", "always_small", "always_large", "cascade"]):
            key_methods[name] = data

    if not key_methods:
        print("No methods to plot for per-bin chart")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    n_methods = len(key_methods)
    x = np.arange(len(bins))
    width = 0.8 / max(n_methods, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for i, (name, data) in enumerate(key_methods.items()):
        per_bin = data.get("aggregate_metrics", {}).get("per_bin", {})
        values = [per_bin.get(b, {}).get("ndcg_at_10", 0) for b in bins]
        short_name = name.split("_seed")[0]
        ax.bar(x + i * width, values, width, label=short_name, color=colors[i])

    ax.set_xlabel("Cold-Start Bin (k)", fontsize=11)
    ax.set_ylabel("NDCG@10", fontsize=11)
    ax.set_title("NDCG@10 by Cold-Start Severity", fontsize=12)
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([f"k={b}" for b in bins])
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_escalation_rate(results: dict, output_path: str):
    """Escalation rate by cold-start bin for cascade methods."""
    bins = ["1", "3", "5", "10", "20"]

    cascade_methods = {
        name: data for name, data in results.items() if "cascade" in name
    }
    if not cascade_methods:
        print("No cascade methods for escalation plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, data in cascade_methods.items():
        per_bin = data.get("aggregate_metrics", {}).get("per_bin", {})
        rates = [per_bin.get(b, {}).get("escalation_rate", 0) * 100 for b in bins]
        ax.plot(bins, rates, "o-", label=name.replace("cascade_", ""), markersize=6)

    ax.set_xlabel("Cold-Start Bin (k)", fontsize=11)
    ax.set_ylabel("Escalation Rate (%)", fontsize=11)
    ax.set_title("Router Escalation Rate by Cold-Start Severity", fontsize=12)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="experiments")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results")
    for name in results:
        agg = results[name].get("aggregate_metrics", {})
        print(f"  {name}: NDCG@10={agg.get('ndcg_at_10', 0):.4f}, "
              f"tokens={agg.get('avg_tokens_per_user', 0):.0f}")

    plot_pareto(results, os.path.join(args.output_dir, "pareto_curve.pdf"))
    plot_perbin(results, os.path.join(args.output_dir, "perbin_heatmap.pdf"))
    plot_escalation_rate(results, os.path.join(args.output_dir, "escalation_rate.pdf"))


if __name__ == "__main__":
    main()
