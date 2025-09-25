from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer


class PGVectorRetriever(BaseModel):
    conn_str: str
    table: str = "chunks"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = "mps"
    top_k: int = 5
    max_k: int = 10

    _model: Optional[SentenceTransformer] = None  # pydantic private ok if not declared as field

    class Config:
        arbitrary_types_allowed = True

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.embed_model_name, device=self.embed_device)

    def _embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        v = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return v

    def _vec_literal(self, v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"

    def retrieve(self, query: str, company: str, year: int, k: Optional[int] = None) -> List[Tuple[float, str]]:
        k = min(self.top_k, self.max_k) if k is None else min(k, self.max_k)
        qvec = self._embed(query)
        qlit = self._vec_literal(qvec)

        sql = f"""
        SELECT text, (embedding <=> { '%s' }::vector) AS dist
        FROM {self.table}
        WHERE company = %s AND year = %s
        ORDER BY embedding <=> { '%s' }::vector
        LIMIT %s;
        """
        # Note: we pass the same vector literal twice (for select and order), parameterized
        with psycopg.connect(self.conn_str) as conn, conn.cursor() as cur:
            cur.execute(sql, (qlit, company, year, qlit, k))
            rows = cur.fetchall()

        # cosine distance: smaller = better; convert to similarity
        out = []
        for text, dist in rows:
            sim = 1.0 - float(dist)
            out.append((sim, text))
        return out
