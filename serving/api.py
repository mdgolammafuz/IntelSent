# serving/api.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.generator import generate_answer
from rag.driver import find_revenue_driver

def create_app() -> FastAPI:
    app = FastAPI(title="IntelSent++ API", version="0.0.1")
    Instrumentator().instrument(app).expose(app)

    class QueryBody(BaseModel):
        text: str
        top_k: int = 5
        use_rerank: bool = True
        company: str | None = None  # optional filter, e.g. "MSFT" or "AAPL"

    @app.on_event("startup")
    def on_startup():
        app.state.retriever = HybridRetriever(alpha=0.7, top_k=50)
        app.state.reranker = CrossReranker()

    def get_retriever() -> HybridRetriever:
        return app.state.retriever

    def get_reranker() -> CrossReranker:
        return app.state.reranker

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/query")
    def query(
        body: QueryBody,
        retr: HybridRetriever = Depends(get_retriever),
        rr: CrossReranker = Depends(get_reranker),
    ):
        # If the question asks about a revenue "driver", try deterministic scan first
        if "driver" in body.text.lower() and "revenue" in body.text.lower():
            direct = find_revenue_driver(company=body.company)
            if direct:
                return {"answer": direct["answer"], "chunks": [direct["chunk"]], "mode": "deterministic-scan"}

        # Otherwise run normal RAG: retrieve -> (optional) rerank -> extract -> generate
        hits = retr.retrieve(body.text)
        docs = retr.docs_from_hits(hits)
        docs = rr.rerank(body.text, docs, top_k=body.top_k) if body.use_rerank else docs[: body.top_k]

        ctx = [d["text"] for d in docs]
        extracted = extract_driver(ctx) if ("revenue" in body.text.lower()) else None
        answer = extracted if extracted else generate_answer(ctx, body.text)
        return {"answer": answer, "chunks": docs, "mode": "rag"}

    @app.get("/")
    def root():
        return {"msg": "IntelSent++ ready. Use /query or /docs."}

    return app

app = create_app()
