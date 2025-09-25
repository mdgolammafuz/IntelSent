
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------
# Utility: load YAML config
# ---------------------------
def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # sensible defaults so missing keys don't crash the app
    cfg.setdefault("artifacts_dir", "artifacts")

    # embedding defaults
    cfg.setdefault("embedding", {})
    cfg["embedding"].setdefault("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    cfg["embedding"].setdefault("device", "mps")

    # retrieval defaults
    cfg.setdefault("retrieval", {})
    cfg["retrieval"].setdefault("top_k", 5)
    cfg["retrieval"].setdefault("max_k", 10)

    # generation defaults
    cfg.setdefault("generation", {})
    cfg["generation"].setdefault("max_context_chars", 6000)
    cfg["generation"].setdefault("model", "gpt-4o-mini")
    cfg["generation"].setdefault("use_openai", True)  # you can disable via CLI/script

    # rerank defaults (not required)
    cfg.setdefault("rerank", {})
    cfg["rerank"].setdefault("enabled", False)
    cfg["rerank"].setdefault("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # pgvector defaults
    cfg.setdefault("pgvector", {})
    cfg["pgvector"].setdefault("enabled", False)
    cfg["pgvector"].setdefault("conn", "postgresql://intel:intel@localhost:5432/intelrag")
    cfg["pgvector"].setdefault("table", "chunks")
    # IMPORTANT: set your actual text column name here (e.g., "content" or "chunk")
    cfg["pgvector"].setdefault("text_col", "content")

    return cfg


# ---------------------------
# Embedder
# ---------------------------
@dataclass
class Embedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dim(self) -> int:
        # trigger a single encode to get the dimensionality
        v = self.encode(["test"])[0]
        return len(v)

    def encode(self, texts: List[str]) -> List[List[float]]:
        import numpy as np

        embs = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embs]


# ---------------------------
# PGVector retriever (configurable text column)
# ---------------------------
@dataclass
class PGVectorRetriever:
    conn_str: str
    table: str = "chunks"
    text_col: str = "content"  # <-- configurable column with the chunk text
    embedder: Optional[Embedder] = None

    def get_relevant(
        self,
        query: str,
        company: Optional[str],
        year: Optional[int],
        k: int,
    ) -> List[Dict[str, Any]]:
        """
        Cosine ANN with pgvector: ORDER BY embedding <-> %s::vector
        We pass the query embedding as a vector parameter.
        """
        if self.embedder is None:
            raise RuntimeError("PGVectorRetriever requires an embedder")

        q_vec = self.embedder.encode([query])[0]  # 384-d for MiniLM

        # Build SQL selecting *configured* text column and alias to 'chunk'
        sql = f"""
        SELECT id, company, year, {self.text_col} AS chunk
        FROM {self.table}
        WHERE (company = COALESCE(%s, company))
          AND (year = COALESCE(%s, year))
        ORDER BY embedding <-> %s
        LIMIT %s;
        """

        import psycopg

        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                # parameters: company, year, query_vector, k
                cur.execute(sql, (company, year, q_vec, k))
                rows = cur.fetchall()

        hits = [
            {
                "id": r[0],
                "company": r[1],
                "year": r[2],
                "chunk": r[3],
            }
            for r in rows
        ]
        return hits


# ---------------------------
# Simple chain
# ---------------------------
@dataclass
class SimpleChain:
    retriever: PGVectorRetriever
    embedder: Embedder
    max_context_chars: int = 6000
    use_openai: bool = False
    openai_model: str = "gpt-4o-mini"

    def _gather_context(
        self, question: str, company: Optional[str], year: Optional[int], top_k: int
    ) -> Tuple[str, List[str]]:
        hits = self.retriever.get_relevant(question, company, year, top_k)
        # build a context string up to max_context_chars
        kept: List[str] = []
        total = 0
        for h in hits:
            chunk = h["chunk"] or ""
            length = len(chunk)
            if total + length > self.max_context_chars:
                break
            kept.append(chunk)
            total += length

        context_text = "\n\n".join(kept)
        return context_text, kept

    def _answer_local(self, question: str, context_text: str) -> str:
        """
        Lightweight 'no-LLM' answerer – return a snippet from the top context,
        so the script can run with --no-openai and still produce something coherent.
        """
        if not context_text:
            return "No relevant context found."
        # Just take the first ~800 chars for display
        return context_text[:800]

    def _answer_openai(self, question: str, context_text: str) -> str:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        if not os.getenv("OPENAI_API_KEY"):
            # safety: fallback to local if key missing
            return self._answer_local(question, context_text)

        llm = ChatOpenAI(model=self.openai_model, temperature=0)
        system = SystemMessage(
            content=(
                "You answer questions strictly using the provided SEC filing context. "
                "If the answer cannot be found in the context, say you don't see it."
            )
        )
        human = HumanMessage(
            content=f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        )
        out = llm.invoke([system, human])
        return out.content.strip()

    # LCEL-compatible method name
    def invoke(
        self, question: str, company: Optional[str], year: Optional[int], top_k: int
    ) -> Dict[str, Any]:
        context_text, kept = self._gather_context(question, company, year, top_k)
        if self.use_openai:
            ans = self._answer_openai(question, context_text)
        else:
            ans = self._answer_local(question, context_text)
        return {
            "answer": ans,
            "contexts": kept,
            "meta": {"company": company, "year": year},
        }

    # convenience for scripts that call .run()
    def run(
        self, question: str, company: Optional[str], year: Optional[int], top_k: int
    ) -> Dict[str, Any]:
        return self.invoke(question, company, year, top_k)


# ---------------------------
# Factory
# ---------------------------
def load_chain(cfg_path: str) -> SimpleChain:
    """
    Creates a SimpleChain using PGVector (as per your current setup).
    """
    cfg = load_config(cfg_path)

    # embedder
    embed_cfg = cfg.get("embedding", {})
    embedder = Embedder(
        model_name=embed_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=embed_cfg.get("device", "mps"),
    )

    # pg retriever
    pg_cfg = cfg.get("pgvector", {})
    retriever = PGVectorRetriever(
        conn_str=pg_cfg.get("conn", "postgresql://intel:intel@localhost:5432/intelrag"),
        table=pg_cfg.get("table", "chunks"),
        text_col=pg_cfg.get("text_col", "content"),
        embedder=embedder,
    )

    # generation
    gen_cfg = cfg.get("generation", {})
    chain = SimpleChain(
        retriever=retriever,
        embedder=embedder,
        max_context_chars=int(gen_cfg.get("max_context_chars", 6000)),
        use_openai=bool(gen_cfg.get("use_openai", True)),
        openai_model=gen_cfg.get("model", "gpt-4o-mini"),
    )
    return chain
