import os
import pickle
import re
from typing import List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Artifact paths
BASE = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH = os.path.join(BASE, "artifacts", "sec_faiss.index")
CHUNKS_PKL = os.path.join(BASE, "artifacts", "chunks.pkl")


def _tok(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w and w.lower() not in ENGLISH_STOP_WORDS]


# Heuristics for driver-style sentences and domain keywords
_REV = re.compile(r"\b(revenue|sales)\b", re.I)
_DRV = re.compile(r"(driven by|led by|primarily by|resulting from|due to|because of)", re.I)
_KEYS = re.compile(
    r"\b(azure|windows|office|microsoft 365|search|advertising|dynamics|cloud|services|xbox|iphone)\b",
    re.I,
)


class HybridRetriever:
    """
    Hybrid retriever with dense + BM25 fusion and heuristic boosts for 'revenue driver' queries.

    - Dense search over a wide pool (FAISS IndexFlatIP; normalized vectors)
    - BM25 over full corpus (+query expansion for driver queries)
    - Candidate set = union(dense_topK, bm25_topK)
    - Fusion score = alpha * bm25_norm + (1-alpha) * dense_norm
    - Heuristic boosts for 'revenue ... driven by ...' (+ keywords)
    """

    def __init__(
        self,
        alpha: float = 0.85,
        top_k: int = 300,
        candidate_pool: int = 1000,
        bm25_pool: int = 1000,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.alpha = alpha
        self.top_k = top_k
        self.candidate_pool = candidate_pool
        self.bm25_pool = bm25_pool

        # Load index and chunks
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PKL, "rb") as f:
            data = pickle.load(f)
        self.chunks: List[str] = data["chunks"]
        self.meta = data["meta"]

        # Models
        self.embedder = SentenceTransformer(model_name)
        self.bm25 = BM25Okapi([_tok(t) for t in self.chunks])

    def _bm25_scores(self, query: str, driver_like: bool) -> np.ndarray:
        """BM25 scores normalized to [0,1]; expand query if driver-like."""
        if driver_like:
            query = query + " revenue increased growth led driven"
        scores = np.array(self.bm25.get_scores(_tok(query)), dtype=np.float32)
        if scores.size == 0:
            return scores
        rng = float(scores.max() - scores.min())
        return (scores - scores.min()) / (rng if rng > 0 else 1.0)

    def _dense_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], normalize_embeddings=True).astype("float32")[0]

    def _dense_topk(self, q_vec: np.ndarray, k: int):
        D, I = self.index.search(q_vec.reshape(1, -1), k)
        return I[0].tolist(), D[0].tolist()

    def _dense_score_for_idx(self, q_vec: np.ndarray, idx: int) -> float:
        """Dense similarity for a specific index using FAISS reconstruct (IndexFlat*)."""
        vec = self.index.reconstruct(idx)
        return float(np.dot(q_vec, np.asarray(vec, dtype=np.float32)))

    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        q_lower = query.lower()
        driver_like = ("driver" in q_lower) and ("revenue" in q_lower)

        total = len(self.chunks)
        k_dense = min(self.candidate_pool, total)

        # Dense pool
        q_vec = self._dense_query(query)
        dense_idx, dense_sim = self._dense_topk(q_vec, k_dense)
        dense_map = {i: s for i, s in zip(dense_idx, dense_sim)}

        # BM25 scores (with driver expansion if applicable)
        bm_norm = self._bm25_scores(query, driver_like)
        bm_top = np.argsort(bm_norm)[::-1][: min(self.bm25_pool, total)].tolist()

        # Candidate union
        cand_set = set(dense_idx) | set(bm_top)

        fused: List[Tuple[int, float]] = []
        for idx in cand_set:
            d_raw = dense_map.get(idx)
            if d_raw is None:
                d_raw = self._dense_score_for_idx(q_vec, idx)
            d_norm = (d_raw + 1.0) / 2.0  # [-1,1] -> [0,1]
            bm = float(bm_norm[idx]) if bm_norm.size else 0.0

            score = self.alpha * bm + (1.0 - self.alpha) * d_norm

            if driver_like:
                t = self.chunks[idx]
                if _REV.search(t) and _DRV.search(t):
                    score += 0.25
                    if _KEYS.search(t):
                        score += 0.10

            fused.append((idx, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: self.top_k]

    def docs_from_hits(self, hits: List[Tuple[int, float]]) -> List[dict]:
        out = []
        for idx, s in hits:
            m = self.meta[idx]
            out.append(
                {
                    "chunk_id": m["chunk_id"],
                    "doc_id": m["doc_id"],
                    "company": m["company"],
                    "year": m["year"],
                    "score": float(s),
                    "text": self.chunks[idx],
                }
            )
        return out
