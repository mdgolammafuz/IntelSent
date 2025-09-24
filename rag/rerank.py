
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

try:
    from sentence_transformers import CrossEncoder  # optional, cached earlier
    HAS_CE = True
except Exception:
    HAS_CE = False


@dataclass
class RerankResult:
    text: str
    score: float
    meta: Dict[str, Any]


class CrossEncoderReranker:
    """
    CrossEncoder-based re-ranker.
    Falls back to a trivial 'keep order' if CrossEncoder isn't available.
    """

    model_name: str
    top_k: int

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        if HAS_CE:
            try:
                self.model = CrossEncoder(self.model_name)
            except Exception:
                self.model = None  # fallback gracefully

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int | None = None) -> List[RerankResult]:
        if top_k is None:
            top_k = self.top_k

        if not docs:
            return []

        # docs expected: [{"text": "...", "meta": {...}}, ...]
        pairs = [(query, d["text"]) for d in docs]

        if self.model is None:
            # Fallback: preserve order, add dummy score
            scored: List[Tuple[float, Dict[str, Any]]] = [(1.0, d) for d in docs]
        else:
            scores = self.model.predict(pairs, batch_size=32, convert_to_numpy=True)
            scored = list(zip(scores, docs))

        scored.sort(key=lambda x: float(x[0]), reverse=True)
        top = scored[: top_k]
        return [
            RerankResult(text=d["text"], score=float(s), meta=d.get("meta", {}))
            for (s, d) in top
        ]


def compress_with_reranker(query: str, docs: List[Dict[str, Any]], reranker: CrossEncoderReranker, k: int) -> List[Dict[str, Any]]:
    """
    'Contextual compression' via re-ranking + cut to top-k.
    Returns the same doc shape used throughout: {"text": str, "meta": {...}}
    """
    ranked = reranker.rerank(query, docs, top_k=k)
    return [{"text": r.text, "meta": r.meta} for r in ranked]
