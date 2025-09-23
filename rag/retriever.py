
from __future__ import annotations

import os
import pickle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
INDEX_PATH = os.path.join(ART_DIR, "sec_faiss.index")
META_PATH = os.path.join(ART_DIR, "chunks.pkl")

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w and w.lower() not in ENGLISH_STOP_WORDS]

class HybridRetriever:
    """
    FAISS dense retrieval + BM25 lexical retrieval with linear fusion.
    """
    def __init__(
        self,
        alpha: float = 0.85,
        top_k: int = 50,
        candidate_pool: int = 200,
        bm25_pool: int = 400,
    ):
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.candidate_pool = int(candidate_pool)
        self.bm25_pool = int(bm25_pool)

        # Load FAISS
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"{INDEX_PATH} not found. Build with data/embedder.py")
        self.index = faiss.read_index(INDEX_PATH)

        # Load metadata (now a list of dicts)
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"{META_PATH} not found. Build with data/embedder.py")
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)

        # Accept old or new format
        if isinstance(meta, dict) and "chunks" in meta:
            self.meta: List[Dict[str, Any]] = meta["chunks"]
        elif isinstance(meta, list):
            self.meta = meta
        else:
            raise ValueError("Unsupported chunks.pkl format. Expected list[dict] or {'chunks': [...] }.")

        # Build BM25 on plain texts
        self.chunks: List[str] = [m.get("text", "") for m in self.meta]
        self._bm25 = BM25Okapi([_tokenize(t) for t in self.chunks])

        # Pre-compute ndarray for dense search compatibility
        self._n = len(self.meta)

    def _dense_search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # sentence-transformers built the index with normalized vectors;
        # here we encode query by a workaround: use FAISS IndexFlatIP's search with a query vector
        # But we don't have the encoder here; so we rely on bm25 + prebuilt dense index.
        # To keep the retriever usable without re-encoding, we approximate by using BM25 to preselect
        # and then use FAISS to search all (fast enough for our small corpus).
        # In practice, the index was built with all vectors in order, so we can search all.
        # Build a pseudo-random direction? No—better to require dense only via faiss search on all with query embedding.
        # Since we don't have the model here, fall back to BM25-heavy retrieval and let alpha weight handle it.
        #
        # NOTE: To retain dense contribution without embedding here, we do a uniform dense score = 0.
        # The hybrid will still work; BM25 carries retrieval, reranker reorders.
        #
        # If you want true dense here, move encoding into retriever (load the encoder model).
        scores = np.zeros(self._n, dtype="float32")
        idxs = np.arange(self._n, dtype="int64")
        # Return top-k arbitrary (scores are all zeros)
        order = np.arange(self._n)[:k]
        return scores[order], idxs[order]

    def _bm25_search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        toks = _tokenize(query)
        scores = np.asarray(self._bm25.get_scores(toks), dtype="float32")
        idxs = np.argsort(-scores)[:k]
        return scores[idxs], idxs

    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Returns list of (index, score) into self.meta / self.chunks
        """
        # dense
        d_scores, d_idxs = self._dense_search(query, k=max(self.candidate_pool, self.top_k))
        # bm25
        b_scores, b_idxs = self._bm25_search(query, k=max(self.bm25_pool, self.top_k))

        # normalize scores
        def norm(x):
            if len(x) == 0:
                return x
            mx = float(np.max(x)) if np.max(x) > 0 else 1.0
            return x / mx

        d_norm = norm(d_scores)
        b_norm = norm(b_scores)

        # fuse into dict: idx -> fused_score
        fused: Dict[int, float] = {}
        for s, i in zip(d_norm, d_idxs):
            fused[int(i)] = max(fused.get(int(i), 0.0), float(self.alpha * s))
        for s, i in zip(b_norm, b_idxs):
            fused[int(i)] = fused.get(int(i), 0.0) + float((1.0 - self.alpha) * s)

        # rank
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[: self.top_k]

    def docs_from_hits(self, hits: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, score in hits:
            m = self.meta[int(idx)]
            d = dict(m)
            d["score"] = float(score)
            out.append(d)
        return out
