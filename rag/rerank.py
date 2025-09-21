from typing import List, Dict
from sentence_transformers import CrossEncoder

class CrossReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Dict], top_k: int = 3) -> List[Dict]:
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs).tolist()
        for d, s in zip(docs, scores):
            d["ce_score"] = float(s)
        docs = sorted(docs, key=lambda x: x["ce_score"], reverse=True)
        return docs[:top_k]
