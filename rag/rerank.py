
from __future__ import annotations

import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class CrossReranker:
    """
    Cross-encoder reranker with batching, truncation, and pair cap.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 32,
        max_pairs: int = 400,
    ):
        self.device = device or _pick_device()
        self.batch_size = int(batch_size)
        self.max_pairs = int(max_pairs)

        # max_length applies to the concatenated pair; CE will truncate automatically.
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []

        # Cap how many pairs we score (huge speed win)
        cand = docs[: min(self.max_pairs, len(docs))]

        # Build pairs; CE handles tokenization+truncation to max_length
        pairs = [(query, d["text"]) for d in cand]

        # Predict in batches on GPU/MPS if available
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False).tolist()

        # Attach scores & sort
        for d, s in zip(cand, scores):
            d["ce_score"] = float(s)
        cand.sort(key=lambda x: x["ce_score"], reverse=True)

        return cand[: top_k]
