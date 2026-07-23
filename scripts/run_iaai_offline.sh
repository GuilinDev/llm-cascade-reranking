#!/usr/bin/env bash
# IAAI-27 offline validation matrix (see IAAI27_PLAN.md).
# Deterministic: --seed drives Ollama sampling; Tier-1 cached per
# (dataset, model, seed) so cascade variants reuse B1's Tier-1 pass.
# Run in MPPR-idle GPU windows. Safe to interrupt/re-run: Tier-1 cache
# persists, completed configs just get a fresh (cheap) replay.
set -euo pipefail
cd "$(dirname "$0")/.."

ML_CACHE="data/cache/movielens_bpr_top100.json"      # retrieval protocol, top-100, rerank head-10
AZ_CACHE="data/cache/amazon_bpr_sampled20.json"      # sampled protocol: target + 19 negatives, rerank all 20
T1_MISTRAL="mistral-small3.2:24b"
T1_QWEN="qwen3.5:27b"
OUT="experiments/iaai"
SEED=42
mkdir -p "$OUT"

# Yield to MPPR (KDD ddl 7/26): pause before each config while its models
# (nemotron / llama-70b) are loaded. Our own models don't match this filter.
yield_to_mppr() {
  while ollama ps 2>/dev/null | grep -qE 'nemotron|llama3\.1:70b'; do
    echo "$(date '+%F %T') MPPR models loaded — pausing 10 min"
    sleep 600
  done
}

run() {
  yield_to_mppr
  echo "$(date '+%F %T') === $*"
  python3 scripts/run_experiment.py "$@" --seed $SEED --max-candidates $MC --output-dir "$OUT"
}

for CACHE in "$ML_CACHE" "$AZ_CACHE"; do
  [ -f "$CACHE" ] || { echo "SKIP missing $CACHE"; continue; }
  # Amazon sampled protocol: LLM reranks the full 20-candidate pool
  MC=10; [[ "$CACHE" == *sampled20* ]] && MC=20

  # B0: instant, no LLM
  run --method no_rerank --cache "$CACHE"

  for T1 in "$T1_MISTRAL" "$T1_QWEN"; do
    # B1 uniform small (populates the shared Tier-1 cache)
    run --method always_small --cache "$CACHE" --tier1-model "$T1"
    # CascadeRec R0(τ=5) + T2-SC (reuses Tier-1 cache; only escalated users cost)
    run --method cascade --router heuristic --router-threshold 5 --tier2 self_consistency \
        --cache "$CACHE" --tier1-model "$T1" --tier2-model "$T1"
    # CascadeRec R0(τ=5) + T2-MP
    run --method cascade --router heuristic --router-threshold 5 --tier2 multipath \
        --cache "$CACHE" --tier1-model "$T1" --tier2-model "$T1"
  done
done

echo "$(date '+%F %T') === IAAI offline matrix complete ==="
