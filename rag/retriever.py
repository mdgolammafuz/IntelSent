
from __future__ import annotations

import os
import math
import pickle
from typing import Any, Dict, List, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer


ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

FAISS_PATH = os.path.join(ARTIFACTS_DIR, "sec_faiss.index")
CHUNKS_PKL = os.path.join(ARTIFACTS_DIR, "chunks.pkl")
CHUNKS_CSV = os.path.join(DATASETS_DIR, "sec_chunks.csv")

# IMPORTANT: keep this identical to data/embedder.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _device_hint() -> str:
    # SentenceTransformer handles device internally; no need to pass unless forcing
    # We keep it simple and let it auto-pick (MPS/CUDA/CPU)
    return "cpu"


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w.lower() not in ENGLISH_STOP_WORDS]


def _rrf(scores_with_ids: List[Tuple[int, float]], k: float = 60.0) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion helper: expects a ranking list (by descending score).
    We convert to rank positions and apply 1/(k + rank).
    """
    # Sort by score descending to get rank order
    sorted_ids = sorted(scores_with_ids, key=lambda x: x[1], reverse=True)
    fused: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(sorted_ids, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """
    Hybrid retriever:
      - Dense: FAISS inner-product search over normalized embeddings
      - Sparse: BM25 on chunk_text
      - Fusion: RRF of top pools
    """

    def __init__(
        self,
        alpha: float = 0.85,
        top_k: int = 50,
        candidate_pool: int = 400,
        bm25_pool: int = 800,
        model_name: str = MODEL_NAME,
    ):
        self.alpha = float(alpha)
        self.top_k = int(top_k)
        self.candidate_pool = int(candidate_pool)
        self.bm25_pool = int(bm25_pool)

        # Load metadata
        if not os.path.exists(CHUNKS_PKL):
            raise FileNotFoundError(f"Missing {CHUNKS_PKL}. Run data/embedder.py first.")
        with open(CHUNKS_PKL, "rb") as f:
            data = pickle.load(f)

        # chunks.pkl is a list[dict]
        if isinstance(data, dict) and "chunks" in data:
            # older format
            data = data["chunks"]
        if not isinstance(data, list):
            raise TypeError("artifacts/chunks.pkl must be a list of dicts.")

        self.meta: List[Dict[str, Any]] = data

        # Optional CSV (not required if pkl already has everything)
        if os.path.exists(CHUNKS_CSV):
            self.df = pd.read_csv(CHUNKS_CSV)
        else:
            # Construct a DataFrame from meta
            self.df = pd.DataFrame(self.meta)

        # Text corpus for BM25
        self.texts: List[str] = [_normalize_text(d.get("text", "")) for d in self.meta]
        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Load FAISS index
        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(f"Missing {FAISS_PATH}. Run data/embedder.py first.")
        self.index = faiss.read_index(FAISS_PATH)

        # Load query encoder
        # Note: must match embedding model used in data/embedder.py; we normalize there,
        # and will normalize query embeddings here as well.
        self.encoder = SentenceTransformer(model_name)

    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = self.encoder.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idxs = self.index.search(q, k)
        # scores: shape (1, k) inner-product (cosine if vectors normalized)
        out: List[Tuple[int, float]] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            out.append((i, float(s)))
        return out

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # top k indices
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Returns list of (chunk_index, fused_score) sorted by fused_score desc.
        """
        query = _normalize_text(query)
        if not query:
            return []

        # Pools
        d_hits = self._dense_search(query, k=self.candidate_pool)
        b_hits = self._bm25_search(query, k=self.bm25_pool)

        # RRF fusion: create rank-based contributions
        d_rrf = _rrf(d_hits, k=60.0)
        b_rrf = _rrf(b_hits, k=60.0)

        # Weighted sum of RRF contributions
        all_ids = set(d_rrf.keys()) | set(b_rrf.keys())
        fused: List[Tuple[int, float]] = []
        for i in all_ids:
            s = self.alpha * d_rrf.get(i, 0.0) + (1.0 - self.alpha) * b_rrf.get(i, 0.0)
            fused.append((i, s))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[: self.top_k]

    def docs_from_hits(self, hits: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, score in hits:
            m = self.meta[idx]
            out.append({
                "chunk_id": m.get("chunk_id", idx),
                "doc_id": m.get("doc_id"),
                "company": m.get("company"),
                "year": m.get("year"),
                "text": m.get("text"),
                "score": float(score),
                # ce_score will be added by the reranker
            })
        return out
