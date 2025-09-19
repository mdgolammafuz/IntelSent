# IntelSent++
RAG for financial documents (SEC 10-K), built from scratch with optional LangChain wrapper.
- Retriever: BM25 + Dense (hybrid), vector DB upgrade-ready (pgvector/FAISS)
- Generator: Flan-T5 (swappable)
- API: FastAPI
- Extras: CI, Docker, K8s (overlays), HF eval submission

## Quick start (no venv for now)
pip install -r requirements.txt
uvicorn serving.api:app --reload
