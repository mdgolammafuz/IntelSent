from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml

from .retriever import FAISSLocalRetriever
from .rerank import CrossEncoderReranker


@dataclass
class AppConfig:
    artifacts_dir: str
    faiss_index: str
    chunks_meta: str

    embed_model: str
    embed_device: str

    top_k: int
    max_k: int

    rerank_enabled: bool
    rerank_model: str
    rerank_top_n: int

    max_context_chars: int


def _join(base: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(base, rel)


def load_config(cfg_path: str) -> AppConfig:
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    artifacts_dir = raw.get("artifacts_dir", "artifacts")
    faiss_index = _join(artifacts_dir, raw.get("faiss_index", "sec_faiss.index"))
    chunks_meta = _join(artifacts_dir, raw.get("chunks_meta", "chunks.pkl"))

    emb = raw.get("embedding", {}) or {}
    retrieval = raw.get("retrieval", {}) or {}
    rerank = raw.get("rerank", {}) or {}
    gen = raw.get("generation", {}) or {}

    return AppConfig(
        artifacts_dir=artifacts_dir,
        faiss_index=faiss_index,
        chunks_meta=chunks_meta,
        embed_model=emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_device=emb.get("device", "auto"),
        top_k=int(retrieval.get("top_k", 5)),
        max_k=int(retrieval.get("max_k", 20)),
        rerank_enabled=bool(rerank.get("enabled", True)),
        rerank_model=rerank.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        rerank_top_n=int(rerank.get("top_n", 5)),
        max_context_chars=int(gen.get("max_context_chars", 4000)),
    )


class SimpleRAGChain:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.retriever = FAISSLocalRetriever(
            faiss_index_path=cfg.faiss_index,
            chunks_meta_path=cfg.chunks_meta,
            embed_model_name=cfg.embed_model,
            embed_device=cfg.embed_device,
            default_top_k=cfg.top_k,
            max_k=cfg.max_k,
        )
        self.reranker: Optional[CrossEncoderReranker] = None
        if cfg.rerank_enabled:
            self.reranker = CrossEncoderReranker(
                model_name=cfg.rerank_model, top_n=cfg.rerank_top_n
            )

    # main API
    def run(
        self,
        question: str,
        company: Optional[str],
        year: Optional[int],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        hits = self.retriever.search(
            query=question, company=company, year=year, top_k=top_k
        )

        if self.reranker and hits:
            hits = self.reranker.rerank(question, hits)

        contexts = [h["text"] for h in hits]
        context_blob = "\n\n---\n\n".join(contexts)
        if len(context_blob) > self.cfg.max_context_chars:
            context_blob = context_blob[: self.cfg.max_context_chars]

        answer = hits[0]["text"] if hits else "No relevant context found."

        return {
            "answer": answer.strip(),
            "contexts": contexts,
            "n_contexts": len(contexts),
            "meta": {"company": company, "year": year},
        }

    # allow callable usage for older callers
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


# ---- backward-compat shim ----
# Some older code may import/instantiate `_SimpleChain`. Make it an alias.
class _SimpleChain(SimpleRAGChain):
    pass


def load_chain(cfg_path: str) -> SimpleRAGChain:
    cfg = load_config(cfg_path)
    return SimpleRAGChain(cfg)
