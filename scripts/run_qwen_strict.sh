#!/usr/bin/env bash
# Rerun all qwen3.5 configs with --strict-output (freeform runs quarantined:
# 97%/60% parse failure — model wrote essays instead of ranked lists).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="experiments/iaai"; SEED=42; Q="qwen3.5:27b"

yield_to_mppr() {
  while ollama ps 2>/dev/null | grep -qE 'nemotron|llama3\.1:70b'; do
    echo "$(date '+%F %T') MPPR models loaded — pausing 10 min"; sleep 600
  done
}
run() { yield_to_mppr; echo "$(date '+%F %T') === $*";
        python3 scripts/run_experiment.py "$@" --seed $SEED --strict-output --output-dir "$OUT"; }

# MovieLens (rerank head-10)
run --method always_small --cache data/cache/movielens_bpr_top100.json --tier1-model "$Q" --max-candidates 10
run --method cascade --router heuristic --router-threshold 5 --tier2 self_consistency \
    --cache data/cache/movielens_bpr_top100.json --tier1-model "$Q" --tier2-model "$Q" --max-candidates 10
run --method cascade --router heuristic --router-threshold 5 --tier2 multipath \
    --cache data/cache/movielens_bpr_top100.json --tier1-model "$Q" --tier2-model "$Q" --max-candidates 10
# Amazon (sampled-20, rerank all 20)
run --method always_small --cache data/cache/amazon_bpr_sampled20.json --tier1-model "$Q" --max-candidates 20
run --method cascade --router heuristic --router-threshold 5 --tier2 self_consistency \
    --cache data/cache/amazon_bpr_sampled20.json --tier1-model "$Q" --tier2-model "$Q" --max-candidates 20
run --method cascade --router heuristic --router-threshold 5 --tier2 multipath \
    --cache data/cache/amazon_bpr_sampled20.json --tier1-model "$Q" --tier2-model "$Q" --max-candidates 20
echo "$(date '+%F %T') === qwen strict rerun complete ==="
