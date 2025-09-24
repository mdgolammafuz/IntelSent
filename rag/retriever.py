from __future__ import annotations

import os
import pickle
from typing import List, Dict, Any, Optional

import faiss  # type: ignore
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.config import ConfigDict
from sentence_transformers import SentenceTransformer


def _pick_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    return "cpu"


class FAISSLocalRetriever(BaseModel):
    # public config fields (these will appear in the schema)
    faiss_index_path: str
    chunks_meta_path: str
    embed_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embed_device: str = Field(default="auto")
    default_top_k: int = Field(default=5)
    max_k: int = Field(default=20)

    # internal runtime objects (not serialized)
    _index: Any = PrivateAttr(default=None)
    _chunks: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _embedder: Optional[SentenceTransformer] = PrivateAttr(default=None)

    # IMPORTANT: allow underscore names
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # <-- this line allows attributes starting with "_"
    )

    def model_post_init(self, __context: Any) -> None:
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        self._index = faiss.read_index(self.faiss_index_path)

        if not os.path.exists(self.chunks_meta_path):
            raise FileNotFoundError(f"Chunks meta not found: {self.chunks_meta_path}")
        with open(self.chunks_meta_path, "rb") as f:
            self._chunks = pickle.load(f)

        device = _pick_device(self.embed_device)
        self._embedder = SentenceTransformer(self.embed_model_name, device=device)

    def _embed(self, texts: List[str]) -> np.ndarray:
        assert self._embedder is not None
        vecs = self._embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        if isinstance(vecs, list):
            vecs = np.array(vecs, dtype="float32")
        return vecs.astype("float32")

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        year: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k = min(self.max_k, top_k or self.default_top_k)
        qv = self._embed([query])
        D, I = self._index.search(qv, k)

        hits: List[Dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0 or idx >= len(self._chunks):
                continue
            meta = dict(self._chunks[idx])
            if company and str(meta.get("company")) != str(company):
                continue
            if year and int(meta.get("year", 0)) != int(year):
                continue
            meta.update(
                {
                    "score": float(dist),
                    "rank": int(rank),
                    "idx": int(idx),
                    "text": meta.get("text", ""),
                }
            )
            hits.append(meta)

        # fallback without company/year filter if strict filter yields nothing
        if not hits and (company or year):
            for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
                if idx < 0 or idx >= len(self._chunks):
                    continue
                meta = dict(self._chunks[idx])
                meta.update(
                    {
                        "score": float(dist),
                        "rank": int(rank),
                        "idx": int(idx),
                        "text": meta.get("text", ""),
                    }
                )
                hits.append(meta)

        return hits[:k]
