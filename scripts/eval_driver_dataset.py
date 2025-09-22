# scripts/eval_driver_dataset.py
import os, sys, json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.generator import generate_answer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "datasets" / "qa_driver.jsonl"

def load_items(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def evaluate(items, alpha=0.85, k_dense=10000, k_bm25=10000, rerank_k=200, top_k_eval=5):
    retr = HybridRetriever(alpha=alpha, top_k=k_dense, candidate_pool=k_dense, bm25_pool=k_bm25)
    rr = CrossReranker()
    tot, r_hits, em_hits = 0, 0, 0

    for ex in items:
        q = ex["question"]
        co = ex["company"]
        gt_chunk = ex["chunk_id"]
        gt_ans = ex["answer"].lower()

        # retrieve wide, rerank deep
        hits = retr.retrieve(q)
        docs = retr.docs_from_hits(hits)
        docs = rr.rerank(q, docs, top_k=min(rerank_k, len(docs)))

        # prefer company's docs if present
        co_docs = [d for d in docs if d["company"] == co] or docs
        # push driver-like sentences first (cheap heuristic)
        driver_like = [d for d in co_docs if ("revenue" in d["text"].lower() and "driven by" in d["text"].lower())]
        if driver_like:
            co_docs = driver_like + [d for d in co_docs if d not in driver_like]

        docs5 = co_docs[:top_k_eval]
        ctx = [d["text"] for d in docs5]
        pred = extract_driver(ctx) or generate_answer(ctx, q)
        pred = (pred or "").lower()

        r_at5 = int(any(d["chunk_id"] == gt_chunk for d in docs5))
        em = int(gt_ans in pred)

        tot += 1; r_hits += r_at5; em_hits += em
        print(f"{co}: GT='{gt_ans}' | Pred='{pred}' | R@{top_k_eval}={r_at5} | EM={em}")

    if tot:
        print(f"\nSummary: Recall@{top_k_eval}={r_hits/tot:.2f} | ExactMatch={em_hits/tot:.2f} on {tot} items")

if __name__ == "__main__":
    items = load_items(DATA)
    if not items:
        print(f"No items found at {DATA}. Run scripts/build_eval_set.py first.")
    else:
        evaluate(items)
