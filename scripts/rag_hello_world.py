import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.generator import generate_answer

if __name__ == "__main__":
    question = "What was a main revenue driver mentioned?"
    retriever = HybridRetriever(alpha=0.7, top_k=50)  # bias toward BM25 keywords; gather more
    hits = retriever.retrieve(question)
    docs = retriever.docs_from_hits(hits)

    reranker = CrossReranker()
    docs = reranker.rerank(question, docs, top_k=5)

    ctx_texts = [d["text"] for d in docs]
    answer = extract_driver(ctx_texts) or generate_answer(ctx_texts, question)

    print("Query:", question)
    print("\nTop Retrieved Chunks (reranked):\n")
    for d in docs:
        print(f"- {d['doc_id']} ({d['company']} {d['year']}) [ce={d['ce_score']:.3f}]: {d['text'][:160]}...")
    print("\nAnswer:\n", answer)
