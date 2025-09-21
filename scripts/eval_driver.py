# scripts/eval_driver.py
import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.driver import find_revenue_driver
from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.generator import generate_answer

COMPANIES = ["MSFT", "AAPL"]

_REV = re.compile(r"\b(revenue|sales)\b", re.I)
_DRV = re.compile(r"(driven by|led by|primarily by|resulting from|due to|because of)", re.I)

def prioritize_driver_sentences(docs, top_n=5):
    driver_docs = [d for d in docs if _REV.search(d["text"]) and _DRV.search(d["text"])]
    seen = set(); out = []
    for d in driver_docs:
        key = (d["doc_id"], d["chunk_id"])
        if key not in seen:
            out.append(d); seen.add(key)
    for d in docs:
        key = (d["doc_id"], d["chunk_id"])
        if key not in seen:
            out.append(d); seen.add(key)
    return out[:top_n]

def main():
    # Pull essentially all chunks from retriever for this small corpus
    retr = HybridRetriever(alpha=0.85, top_k=10000, candidate_pool=10000, bm25_pool=10000)
    rr = CrossReranker()
    results = []
    q = "Which product or service was revenue growth driven by?"

    for co in COMPANIES:
        gt = find_revenue_driver(company=co)
        if not gt:
            print(f"[skip] no ground truth for {co}")
            continue

        # 1) Retrieve EVERYTHING (within top_k safety)
        hits = retr.retrieve(q)
        docs = retr.docs_from_hits(hits)

        # 2) Rerank ALL candidates (don’t cut early)
        docs = rr.rerank(q, docs, top_k=len(docs))

        # 3) Prefer the company's docs if available
        company_docs = [d for d in docs if d["company"] == co] or docs

        # 4) Prioritize driver-style sentences, then keep 5
        docs = prioritize_driver_sentences(company_docs, top_n=5)

        # 5) Predict
        ctx = [d["text"] for d in docs]
        pred = extract_driver(ctx) or generate_answer(ctx, q)

        # 6) Metrics
        r_at5 = int(any(d["chunk_id"] == gt["chunk"]["chunk_id"] for d in docs))
        em = int(gt["answer"] in (pred or "").lower())
        print(f"{co}: GT='{gt['answer']}' | Pred='{pred}' | R@5={r_at5} | EM={em}")
        results.append((r_at5, em))

    if results:
        r5 = sum(r for r, _ in results) / len(results)
        em = sum(e for _, e in results) / len(results)
        print(f"\nSummary: Recall@5={r5:.2f} | ExactMatch={em:.2f} on {len(results)} items")

if __name__ == "__main__":
    main()
