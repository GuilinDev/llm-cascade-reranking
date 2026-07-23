# LLM Cascade Reranking

**Adaptive LLM compute allocation for personalized reranking: a small model reranks every request, a router escalates only the requests where expensive inference is likely to help.**

> **Paper:** *CascadeRec: Deployed Adaptive LLM Compute Allocation for Multi-Tenant Content Recommendation* (under review, IAAI-27). The system is referred to as **CascadeRec** in the paper.

This repository contains the complete offline validation pipeline from the paper: datasets, first-stage retrieval, LLM reranking tiers, routers, metrics, statistical analysis, and every cached artifact needed to reproduce the reported tables. The production deployment (AWS EKS / GKE, vLLM + Amazon Bedrock) is described in the paper; tenant data and production code are not released.

---

## Repository layout

```
src/
  datasets/    MovieLens / Amazon loaders, cold-start truncation & binning
  baselines/   first-stage retrievers (numpy iALS with fold-in, BM25, popularity)
  llm/         Ollama client (streaming, hard timeouts, seeded sampling)
  cascade/     Tier-1/Tier-2 rerankers, prompts, parsing, Borda aggregation,
               Tier-1 output cache, cascade orchestrator + baselines
  routers/     R0 heuristic, R1 self-confidence, R2 logistic, R3 LLM-coordinator
  metrics/     NDCG/Recall/MRR + per-bin aggregation, parse-failure accounting
scripts/       runnable entry points (see below)
data/cache/    first-stage candidate caches incl. sampled user IDs (committed)
data/cache/tier1/  cached Tier-1 LLM outputs per (dataset, model, seed)
experiments/iaai/  raw result JSONs behind every table in the paper
tests/         unit tests for ranking metrics
```

## Requirements

- Python 3.10+, `pip install -r requirements.txt`
- [Ollama](https://ollama.com) serving `mistral-small3.2:24b` and `qwen3.5:27b`
  (all experiments ran on a single NVIDIA DGX Spark; any ~24–27B-capable GPU works)

## Reproducing the paper's tables

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Data (only needed to rebuild caches from scratch — committed caches skip this)
bash scripts/download_movielens.sh                  # MovieLens-1M -> data/raw/ml-1m
# Amazon Electronics 2023 downloads automatically on first cache build (~8 GB)

# 3. Candidate caches (committed under data/cache/; rebuild if desired)
python scripts/cache_candidates.py --dataset movielens --retriever bpr --topk 100 \
    --cold-start-ks 1,3,5,10,20 --max-users-per-k 200
python scripts/cache_candidates.py --dataset amazon --retriever bpr \
    --candidate-protocol sampled --n-negatives 19 --max-reviews 15000000 \
    --min-item-interactions 20 --cold-start-ks 1,3,5,10,20 --max-users-per-k 200

# 4. Full experiment matrix (2 datasets x 2 models x {B0, B1, CR+SC, CR+MP})
bash scripts/run_iaai_offline.sh          # Mistral configs (freeform prompts)
bash scripts/run_qwen_strict.sh           # Qwen configs (--strict-output prompts)

# 5. Statistics + LaTeX tables
python scripts/analyze_iaai.py            # paired bootstrap CIs + Wilcoxon vs B0
python scripts/fill_tables_iaai.py        # emits every table row in the paper
```

Determinism: `--seed` drives Ollama's sampler, and Tier-1 outputs are cached per
(dataset, model, seed) under `data/cache/tier1/`, so all cascade variants reuse one
Tier-1 pass and non-escalated users match the uniform baseline exactly. With the
committed caches, steps 4–5 replay Tier-1 instantly and only re-run Tier-2 calls.

## Candidate protocols (disclosed in the paper)

- **MovieLens-1M** (3.9k items): full-catalog iALS retrieval, top-100 cached, the
  LLM reranks the head-10; retriever Recall@100 is the reranking ceiling.
- **Amazon-Electronics** (104k items after 20-core filtering): target + 19 sampled
  negatives ordered by retriever score, the LLM reranks all 20 (standard sampled
  protocol for LLM ranking on large catalogs).

## A note on `quarantine_qwen_freeform/`

`experiments/iaai/quarantine_qwen_freeform/` preserves the initial Qwen3.5 runs in
which the model answered the reranking prompt with free-text essays instead of the
requested ID list (97% parse failure on MovieLens), silently degrading to retrieval
order. These runs are excluded from the paper's tables and kept as evidence for the
output-compliance lesson (paper, Lessons Learned); the `--strict-output` prompt
suffix that fixes it is in `src/cascade/reranker.py`.

## Citing

```bibtex
@inproceedings{cascaderec2027,
  title  = {CascadeRec: Deployed Adaptive LLM Compute Allocation for
            Multi-Tenant Content Recommendation},
  author = {Zhang, Guilin and Zhao, Kai and Chu, Xu},
  note   = {Under review},
  year   = {2027}
}
```

## License

MIT for code. MovieLens-1M is provided by GroupLens under its own terms of use;
Amazon Reviews 2023 by McAuley Lab (UCSD). Neither dataset is redistributed here —
only derived candidate caches for the 1,000 evaluation users per dataset.
