import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.retriever import HybridRetriever
from rag.generator import generate_answer

if __name__ == "__main__":
    question = "What was a main revenue driver mentioned?"
    retriever = HybridRetriever(alpha=0.5, top_k=3)
    hits = retriever.retrieve(question)
    docs = retriever.docs_from_hits(hits)

    print("Query:", question)
    print("\nTop Retrieved Chunks:\n")
    for d in docs:
        print(f"- {d['doc_id']} ({d['company']} {d['year']}) [{d['score']:.3f}]: {d['text'][:160]}...")

    answer = generate_answer([d["text"] for d in docs], question)
    print("\nAnswer:\n", answer)
