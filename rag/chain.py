
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer


# ---------- tiny config helpers ----------
def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_chain(cfg_path: str) -> "_SimpleChain":
    cfg = _load_yaml(cfg_path)

    # Embedding
    embed_model = _get(cfg, "embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2")
    embed_device = _get(cfg, "embedding.device", "cpu")  # "mps" ok on Apple

    # pgvector
    pg_dsn = _get(cfg, "pgvector.dsn", "postgresql://intel:intel@localhost:5432/intelrag")
    table = _get(cfg, "pgvector.table", "chunks")
    text_col_pref = _get(cfg, "pgvector.text_column", None)  # if you know it, set it; else autodetect

    # generation
    max_context_chars = int(_get(cfg, "generation.max_context_chars", 6000))

    # retrieval extras
    rerank_keep = int(_get(cfg, "retrieval.rerank_keep", 5))

    retriever = PGVectorRetriever(
        dsn=pg_dsn,
        table=table,
        text_col_pref=text_col_pref,
        embed_model=embed_model,
        embed_device=embed_device,
    )

    chain = _SimpleChain(
        retriever=retriever,
        max_context_chars=max_context_chars,
        rerank_keep=rerank_keep,
        use_openai=_env_has_openai() and not _env_force_no_openai(),
    )
    return chain


def _env_has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _env_force_no_openai() -> bool:
    return os.environ.get("NO_OPENAI", "").lower() in {"1", "true", "yes"}


# ---------- pgvector retriever ----------
class PGVectorRetriever:
    """
    - Embeds query with SentenceTransformers
    - Filters by (company, year) when provided
    - Uses <-> with a vector literal cast (%s::vector)
    - Auto-detects the text column if not provided
    """

    _CANDIDATE_TEXT_COLS = ["chunk", "content", "text", "passage", "body"]

    def __init__(
        self,
        dsn: str,
        table: str,
        text_col_pref: Optional[str],
        embed_model: str,
        embed_device: str = "cpu",
    ) -> None:
        self.dsn = dsn
        self.table = table
        self._embedder = SentenceTransformer(embed_model, device=embed_device)

        # figure out the text column once
        self.text_col = self._resolve_text_column(text_col_pref)

    # -- schema probing --
    def _resolve_text_column(self, preferred: Optional[str]) -> str:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (self.table,),
            )
            cols = [r[0] for r in cur.fetchall()]

        # if user specified explicitly and it exists, use it
        if preferred and preferred in cols:
            return preferred

        # try common names
        for c in self._CANDIDATE_TEXT_COLS:
            if c in cols:
                return c

        # nothing matched -> helpful error
        raise RuntimeError(
            f"Could not determine text column for table '{self.table}'. "
            f"Tried {self._CANDIDATE_TEXT_COLS} plus config 'pgvector.text_column'. "
            f"Existing columns: {cols}. "
            f"Fix by either renaming your column to one of the candidates, "
            f"or set pgvector.text_column in config/app.yaml."
        )

    # -- embedding helpers --
    def _embed(self, text: str) -> List[float]:
        vec = self._embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vec.astype(np.float32).tolist()

    @staticmethod
    def _to_pgvector_literal(vec: List[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    # -- public API --
    def get_relevant(
        self, question: str, company: Optional[str], year: Optional[int], top_k: int
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            top_k = 5

        q_lit = self._to_pgvector_literal(self._embed(question))

        where_parts: List[str] = []
        params: List[Any] = [q_lit]  # first param for subquery
        if company is not None:
            where_parts.append("company = %s")
            params.append(company)
        if year is not None:
            where_parts.append("year = %s")
            params.append(int(year))  # ensure int

        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

        sql = f"""
        SELECT id, company, year, {self.text_col} AS chunk, dist
        FROM (
            SELECT id, company, year, {self.text_col},
                   embedding <-> %s::vector AS dist
            FROM {self.table}
            {where_sql}
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ) sub
        ORDER BY dist ASC;
        """
        # we need the same vector again for ORDER BY expression
        exec_params = params + [q_lit, top_k]

        out: List[Dict[str, Any]] = []
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, exec_params)
            for row in cur.fetchall():
                _id, _company, _year, _chunk, _dist = row
                out.append(
                    {
                        "id": _id,
                        "company": _company,
                        "year": _year,
                        "chunk": _chunk,
                        "distance": float(_dist),
                    }
                )
        return out


# ---------- simple chain ----------
class _SimpleChain:
    def __init__(
        self,
        retriever: PGVectorRetriever,
        max_context_chars: int = 6000,
        rerank_keep: int = 5,
        use_openai: bool = False,
    ) -> None:
        self.retriever = retriever
        self.max_context_chars = max_context_chars
        self.rerank_keep = max(rerank_keep, 1)
        self.use_openai = use_openai
        self._openai_chat = None  # lazy

    def run(
        self, question: str, company: Optional[str], year: Optional[int], top_k: int
    ) -> Dict[str, Any]:
        return self.invoke(question, company, year, top_k)

    def invoke(
        self, question: str, company: Optional[str], year: Optional[int], top_k: int
    ) -> Dict[str, Any]:
        hits = self.retriever.get_relevant(question, company, year, top_k)
        hits = hits[: self.rerank_keep]
        context_text, kept = self._pack_context(hits, self.max_context_chars)

        if self.use_openai:
            answer = self._answer_with_openai(question, context_text)
        else:
            answer = self._answer_locally(question, kept)

        return {"answer": answer, "contexts": [h["chunk"] for h in kept], "meta": {"company": company, "year": year}}

    def _pack_context(
        self, hits: List[Dict[str, Any]], max_chars: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        buf, kept, cur = [], [], 0
        for h in hits:
            t = h.get("chunk", "") or ""
            if cur + len(t) + 2 > max_chars:
                break
            buf.append(t)
            kept.append(h)
            cur += len(t) + 2
        return ("\n\n".join(buf), kept)

    def _answer_locally(self, question: str, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "I couldn't find relevant context locally."
        return hits[0]["chunk"]

    def _answer_with_openai(self, question: str, context_text: str) -> str:
        try:
            from openai import OpenAI
        except Exception:
            return self._answer_locally(question, [])

        if self._openai_chat is None:
            self._openai_chat = OpenAI()

        sys_prompt = (
            "You are a helpful assistant answering questions strictly from the provided context. "
            "If the answer isn't present, say you don't know."
        )
        user_prompt = f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"

        try:
            resp = self._openai_chat.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM fallback) {self._answer_locally(question, [])}  // Error: {e}"
