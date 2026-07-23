"""
First-stage retrievers for CascadeRec.

These produce the top-N candidate list per user that CascadeRec's LLM
rerankers operate on. Keeping them lightweight (pure numpy/scipy + optional
`implicit`) so that day-1 can run without heavy infra.

Retrievers implemented:
  - PopularityRetriever    (trivial, strong cold-start baseline)
  - BPRMFRetriever         (matrix factorization via `implicit` if available,
                            else falls back to sparse SVD)
  - BM25Retriever          (uses rank_bm25 for text-side retrieval, used on LaMP-3
                            and as a text fallback on Amazon)

All retrievers expose the same `.top_k(user_id, k) -> List[str]` interface and
support a `.fit(samples, catalog)` step.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Protocol, Sequence, Set

import numpy as np
from scipy import sparse

from ..datasets.base import ItemRecord, UserEvalSample

logger = logging.getLogger(__name__)


class FirstStageRetriever(Protocol):
    """Common interface for first-stage candidate generators."""

    def fit(
        self,
        samples: Sequence[UserEvalSample],
        catalog: Dict[str, ItemRecord],
    ) -> None: ...

    def top_k(self, sample: UserEvalSample, k: int = 100) -> List[str]: ...


class PopularityRetriever:
    """
    Global popularity baseline.

    Ranks items by how many users interacted with them in `samples`.
    Crucial cold-start baseline — for users with zero history, this is the
    only reasonable fallback.
    """

    def __init__(self) -> None:
        self._ranked_items: List[str] = []
        self._item_set: Set[str] = set()

    def fit(
        self,
        samples: Sequence[UserEvalSample],
        catalog: Dict[str, ItemRecord],
    ) -> None:
        counts: Counter = Counter()
        for s in samples:
            for h in s.history:
                iid = h.get("item_id")
                if iid:
                    counts[iid] += 1
        self._ranked_items = [iid for iid, _ in counts.most_common()]
        # pad with remaining catalog for long-tail coverage
        seen = set(self._ranked_items)
        for iid in catalog:
            if iid not in seen:
                self._ranked_items.append(iid)
        self._item_set = set(self._ranked_items)
        logger.info(f"PopularityRetriever fit: {len(self._ranked_items)} items ranked")

    def top_k(self, sample: UserEvalSample, k: int = 100) -> List[str]:
        already_seen = {h["item_id"] for h in sample.history}
        out: List[str] = []
        for iid in self._ranked_items:
            if iid in already_seen:
                continue
            out.append(iid)
            if len(out) >= k:
                break
        return out


class BPRMFRetriever:
    """
    Implicit-feedback matrix factorization first-stage.

    Priority order (first successful wins):
      1. `implicit` library (BPR) — best quality but needs C extension
      2. Hand-rolled numpy iALS (Hu/Koren/Volinsky 2008, ICDM'08)
      3. Truncated SVD on the binary matrix — last-resort fallback

    Cold-start behavior: users absent from training (no history) fall back to
    the global mean user vector — degenerates to popularity ordering.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 20,
        alpha: float = 40.0,  # iALS confidence scaling
        use_implicit: bool = True,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.use_implicit = use_implicit
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []
        self._uid_to_idx: Dict[str, int] = {}
        self._iid_to_idx: Dict[str, int] = {}
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self._global_user_vec: Optional[np.ndarray] = None

    def fit(
        self,
        samples: Sequence[UserEvalSample],
        catalog: Dict[str, ItemRecord],
    ) -> None:
        # build index
        self.user_ids = [s.user_id for s in samples]
        self._uid_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_set: Set[str] = set(catalog.keys())
        for s in samples:
            for h in s.history:
                iid = h.get("item_id")
                if iid:
                    item_set.add(iid)
        self.item_ids = sorted(item_set)
        self._iid_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        if n_users == 0 or n_items == 0:
            logger.warning("BPRMFRetriever.fit called with empty samples/catalog")
            return

        # build sparse user-item matrix using history as positives
        rows, cols, data = [], [], []
        for s in samples:
            u_idx = self._uid_to_idx[s.user_id]
            for h in s.history:
                iid = h.get("item_id")
                if iid in self._iid_to_idx:
                    rows.append(u_idx)
                    cols.append(self._iid_to_idx[iid])
                    data.append(1.0)
        mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        fitted = False
        # --- attempt 1: implicit library ---
        if self.use_implicit:
            try:
                import implicit  # type: ignore

                model = implicit.bpr.BayesianPersonalizedRanking(
                    factors=self.factors,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    use_gpu=False,
                )
                model.fit(mat.T.tocsr(), show_progress=False)
                self.user_factors = np.asarray(model.user_factors)
                self.item_factors = np.asarray(model.item_factors)
                logger.info(
                    f"BPRMFRetriever (implicit BPR) fit: users={n_users}, items={n_items}"
                )
                fitted = True
            except Exception as exc:
                logger.info(f"implicit unavailable ({exc}); trying hand-rolled iALS")

        # --- attempt 2: numpy iALS (Hu/Koren/Volinsky 2008) ---
        if not fitted:
            try:
                U_f, V_f = self._fit_ials(mat)
                self.user_factors = U_f
                self.item_factors = V_f
                logger.info(
                    f"BPRMFRetriever (numpy iALS) fit: users={n_users}, items={n_items}, "
                    f"factors={self.factors}, iters={self.iterations}"
                )
                fitted = True
            except Exception as exc:
                logger.warning(f"numpy iALS failed ({exc}); falling back to truncated SVD")

        # --- attempt 3: truncated SVD ---
        if not fitted:
            from scipy.sparse.linalg import svds

            k = min(self.factors, min(mat.shape) - 1)
            if k < 2:
                k = 2
            try:
                U, S, Vt = svds(mat.astype(np.float32), k=k)
                scale = np.sqrt(S)
                self.user_factors = U * scale
                self.item_factors = (Vt.T) * scale
            except Exception as svd_err:
                logger.warning(f"SVD fallback failed ({svd_err}); degenerating to popularity")
                pop = np.asarray(mat.sum(axis=0)).ravel()
                self.user_factors = np.ones((n_users, 1), dtype=np.float32)
                self.item_factors = pop.reshape(-1, 1).astype(np.float32)
            logger.info(
                f"BPRMFRetriever (SVD fallback) fit: users={n_users}, items={n_items}, "
                f"k={self.user_factors.shape[1]}"
            )

        self._global_user_vec = self.user_factors.mean(axis=0)

    def _fit_ials(self, mat: sparse.csr_matrix):
        """
        Hand-rolled implicit ALS.

        Reference:
            Hu, Koren, Volinsky. "Collaborative Filtering for Implicit
            Feedback Datasets." ICDM 2008.

        Implements the standard iALS alternating-least-squares updates with
        confidence c_{ui} = 1 + alpha * r_{ui} on observed entries. Uses the
        YtY trick to avoid rebuilding the full confidence matrix on each step.
        """
        n_users, n_items = mat.shape
        k = self.factors
        reg = self.regularization
        rng = np.random.default_rng(42)

        X = rng.normal(scale=0.01, size=(n_users, k)).astype(np.float32)
        Y = rng.normal(scale=0.01, size=(n_items, k)).astype(np.float32)

        mat_csr = mat.tocsr().astype(np.float32)
        mat_csc = mat.tocsc().astype(np.float32)
        I = reg * np.eye(k, dtype=np.float32)

        for it in range(self.iterations):
            # --- update user factors X given Y ---
            YtY = Y.T @ Y  # (k, k)
            for u in range(n_users):
                start, end = mat_csr.indptr[u], mat_csr.indptr[u + 1]
                if start == end:
                    continue
                idx = mat_csr.indices[start:end]
                r = mat_csr.data[start:end]  # binary (1 for observed)
                c = 1.0 + self.alpha * r  # confidence
                Y_u = Y[idx]  # (nu, k)
                # A = YtY + Y_u.T * (c-1) * Y_u + reg*I
                A = YtY + (Y_u.T * (c - 1.0)) @ Y_u + I
                b = (Y_u.T * c) @ np.ones_like(r)
                X[u] = np.linalg.solve(A, b)

            # --- update item factors Y given X ---
            XtX = X.T @ X
            for i in range(n_items):
                start, end = mat_csc.indptr[i], mat_csc.indptr[i + 1]
                if start == end:
                    continue
                idx = mat_csc.indices[start:end]
                r = mat_csc.data[start:end]
                c = 1.0 + self.alpha * r
                X_i = X[idx]
                A = XtX + (X_i.T * (c - 1.0)) @ X_i + I
                b = (X_i.T * c) @ np.ones_like(r)
                Y[i] = np.linalg.solve(A, b)

        return X, Y

    def _fold_in_user(self, history: List[Dict]) -> np.ndarray:
        """
        Compute a fresh user factor from an arbitrary history using one iALS
        step with the trained item factors held fixed. Matches the inductive
        cold-start protocol used by LaMP / standard cold-start CF benchmarks.

        This is the mechanism that lets CascadeRec evaluate on artificially
        truncated users: the user's identity (used for leave-last-positive)
        is preserved, but the first-stage re-solves the user vector from only
        the k observed items.
        """
        if self.item_factors is None:
            raise RuntimeError("Retriever not fitted")
        obs_idxs = [
            self._iid_to_idx[h["item_id"]]
            for h in history
            if h.get("item_id") in self._iid_to_idx
        ]
        if not obs_idxs:
            return self._global_user_vec  # genuinely cold — no history signal
        Y_obs = self.item_factors[obs_idxs]  # (n_obs, k)
        k = self.item_factors.shape[1]
        c = 1.0 + self.alpha  # confidence for binary observed items
        A = Y_obs.T @ Y_obs * (c - 1.0) + (self.item_factors.T @ self.item_factors) + self.regularization * np.eye(k)
        b = Y_obs.T @ (np.ones(len(obs_idxs)) * c)
        return np.linalg.solve(A, b)

    def top_k(self, sample: UserEvalSample, k: int = 100) -> List[str]:
        if self.item_factors is None or self.user_factors is None:
            return []
        # Prefer fold-in whenever the sample is a truncated one, OR whenever
        # the user_id isn't in the trained index.
        truncated = sample.extra.get("truncated_to") is not None
        uidx = self._uid_to_idx.get(sample.user_id)
        if truncated or uidx is None:
            user_vec = self._fold_in_user(sample.history)
        else:
            user_vec = self.user_factors[uidx]
        scores = self.item_factors @ user_vec
        # mask out already-seen items
        seen_idxs = {
            self._iid_to_idx[h["item_id"]]
            for h in sample.history
            if h["item_id"] in self._iid_to_idx
        }
        if seen_idxs:
            scores = scores.copy()
            for idx in seen_idxs:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self.item_ids[i] for i in top_idx]

    def rank_items(self, sample: UserEvalSample, item_ids: List[str]) -> List[str]:
        """
        Score an explicit candidate subset with the (fold-in) user vector and
        return it ordered best-first. Used by the sampled-negatives protocol
        (target + N random negatives), where full-catalog retrieval ceilings
        are too low for reranking evaluation (e.g. Amazon).
        Unknown item ids sort last, in input order.
        """
        if self.item_factors is None:
            return list(item_ids)
        truncated = sample.extra.get("truncated_to") is not None
        uidx = self._uid_to_idx.get(sample.user_id)
        if truncated or uidx is None:
            user_vec = self._fold_in_user(sample.history)
        else:
            user_vec = self.user_factors[uidx]
        scored = []
        for pos, iid in enumerate(item_ids):
            idx = self._iid_to_idx.get(iid)
            s = float(self.item_factors[idx] @ user_vec) if idx is not None else -np.inf
            scored.append((-s, pos, iid))
        scored.sort()
        return [iid for _, _, iid in scored]


class BM25Retriever:
    """
    BM25 retrieval over item titles/descriptions.

    Builds a BM25 index over item text, then queries with user-history text
    (concatenation of recent item titles). Intended for LaMP-3 and Amazon where
    textual signal is rich.
    """

    def __init__(self) -> None:
        self.items: Dict[str, ItemRecord] = {}
        self.item_ids: List[str] = []
        self._bm25 = None

    def fit(
        self,
        samples: Sequence[UserEvalSample],
        catalog: Dict[str, ItemRecord],
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank_bm25 is required for BM25Retriever. "
                "pip install rank_bm25"
            ) from exc
        self.items = catalog
        self.item_ids = list(catalog.keys())
        tokenized = []
        for iid in self.item_ids:
            item = catalog[iid]
            doc = f"{item.title} {item.category} {item.description}".lower().split()
            tokenized.append(doc or ["<empty>"])
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25Retriever fit: {len(self.item_ids)} items indexed")

    def _query_tokens(self, sample: UserEvalSample) -> List[str]:
        parts: List[str] = []
        for h in sample.history[-20:]:
            iid = h.get("item_id")
            if iid and iid in self.items:
                parts.append(self.items[iid].title)
                parts.append(self.items[iid].category)
        if sample.input_text:
            parts.append(sample.input_text)
        query = " ".join(parts).lower().split()
        if not query:
            query = ["<empty>"]
        return query

    def top_k(self, sample: UserEvalSample, k: int = 100) -> List[str]:
        if self._bm25 is None:
            return []
        query = self._query_tokens(sample)
        scores = self._bm25.get_scores(query)
        top_idx = np.argsort(-scores)[: k * 2]
        seen = {h["item_id"] for h in sample.history}
        out: List[str] = []
        for i in top_idx:
            iid = self.item_ids[i]
            if iid in seen:
                continue
            out.append(iid)
            if len(out) >= k:
                break
        return out
