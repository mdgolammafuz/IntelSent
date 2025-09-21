from fastapi import FastAPI, Depends
from pydantic import BaseModel
from rag.retriever import HybridRetriever
from rag.generator import generate_answer
from prometheus_fastapi_instrumentator import Instrumentator

def create_app() -> FastAPI:
    app = FastAPI(title="IntelSent++ API", version="0.0.1")

    # Add middleware BEFORE startup (fix for “Cannot add middleware after start”)
    Instrumentator().instrument(app).expose(app)

    class Query(BaseModel):
        text: str
        top_k: int = 3

    @app.on_event("startup")
    def startup():
        app.state.retriever = HybridRetriever(alpha=0.5, top_k=3)

    def get_retriever() -> HybridRetriever:
        return app.state.retriever

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.post("/query")
    def query(q: Query, retr: HybridRetriever = Depends(get_retriever)):
        hits = retr.retrieve(q.text)
        docs = retr.docs_from_hits(hits)
        ctx = [d["text"] for d in docs][: q.top_k]
        ans = generate_answer(ctx, q.text)
        return {"answer": ans, "chunks": docs[: q.top_k]}

    @app.get("/")
    def root():
        return {"msg": "IntelSent++ ready. Use /query or /docs."}

    return app

app = create_app()
