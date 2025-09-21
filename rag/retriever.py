import os, pickle
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BASE = os.path.dirname(os.path.dirname(__file__))
INDEX_PATH = os.path.join(BASE, "artifacts", "sec_faiss.index")
CHUNKS_PKL = os.path.join(BASE, "artifacts", "chunks.pkl")

def _tok(t: str):
    return [w.lower() for w in t.split() if w and w.lower() not in ENGLISH_STOP_WORDS]

class HybridRetriever:
    def __init__(self, alpha: float = 0.5, top_k: int = 3):
        self.alpha = alpha
        self.top_k = top_k
        self.index = faiss.read_index(INDEX_PATH)
        data = pickle.load(open(CHUNKS_PKL, "rb"))
        self.chunks = data["chunks"]
        self.meta = data["meta"]
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.bm25 = BM25Okapi([_tok(t) for t in self.chunks])

    def retrieve(self, query: str) -> List[Tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, min(50, len(self.chunks)))
        I = I[0].tolist(); dense = D[0].tolist()

        bm = self.bm25.get_scores(_tok(query))
        bm = (np.array(bm) - np.min(bm)) / (np.ptp(bm) + 1e-9)

        scored = []
        for idx, d in zip(I, dense):
            d = (d + 1.0) / 2.0
            scored.append((idx, float(self.alpha * bm[idx] + (1 - self.alpha) * d)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]

    def docs_from_hits(self, hits):
        rows = []
        for idx, s in hits:
            m = self.meta[idx]
            rows.append({"chunk_id": m["chunk_id"], "doc_id": m["doc_id"],
                         "company": m["company"], "year": m["year"],
                         "score": s, "text": self.chunks[idx]})
        return rows
