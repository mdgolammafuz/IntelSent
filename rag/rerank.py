from __future__ import annotations

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, PrivateAttr
from pydantic.config import ConfigDict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(BaseModel):
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_n: int = Field(default=5)

    _model: Optional[CrossEncoder] = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # allow underscore private attrs
    )

    def model_post_init(self, __context: Any) -> None:
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return docs
        assert self._model is not None
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self._model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs_sorted = sorted(
            docs, key=lambda x: x.get("rerank_score", 0.0), reverse=True
        )
        return docs_sorted[: min(self.top_n, len(docs_sorted))]
