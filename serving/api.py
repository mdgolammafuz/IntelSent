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

    # Expose /metrics before startup
    Instrumentator().instrument(app).expose(app)

    class QueryBody(BaseModel):
        text: str
        top_k: int = 5
        use_rerank: bool = True
        company: str | None = None  # optional filter for both paths

    class DriverBody(BaseModel):
        company: str | None = None  # e.g., "MSFT", "AAPL"

    @app.on_event("startup")
    def on_startup():
        # Wider candidate set; alpha>0.5 biases BM25 keywords
        app.state.retriever = HybridRetriever(alpha=0.7, top_k=50)
        app.state.reranker = CrossReranker()

    def get_retriever() -> HybridRetriever:
        return app.state.retriever

    def get_reranker() -> CrossReranker:
        return app.state.reranker

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/driver")
    def driver(body: DriverBody):
        """
        Deterministic corpus-wide scan for 'revenue ... driven by ...' sentences.
        Returns a short phrase and the supporting chunk, or 'not found'.
        """
        res = find_revenue_driver(company=body.company)
        if not res:
            return {"answer": "not found", "chunks": [], "mode": "deterministic-scan"}
        return {"answer": res["answer"], "chunks": [res["chunk"]], "mode": "deterministic-scan"}

    @app.post("/query")
    def query(
        body: QueryBody,
        retr: HybridRetriever = Depends(get_retriever),
        rr: CrossReranker = Depends(get_reranker),
    ):
        """
        General RAG:
        - If looks like a 'revenue driver' query, try deterministic scan first (honors company).
        - Else: retrieve -> optional rerank -> (strict company filter if provided) ->
                extract short phrase if applicable -> generator fallback.
        """
        lower_q = body.text.lower()

        if "driver" in lower_q and "revenue" in lower_q:
            direct = find_revenue_driver(company=body.company)
            if direct:
                return {"answer": direct["answer"], "chunks": [direct["chunk"]], "mode": "deterministic-scan"}

        hits = retr.retrieve(body.text)
        docs = retr.docs_from_hits(hits)

        # Optional rerank
        docs = rr.rerank(body.text, docs, top_k=max(body.top_k * 6, body.top_k)) if body.use_rerank else docs

        # STRICT company filter if provided
        if body.company:
            filt = [d for d in docs if d["company"].lower() == body.company.lower()]
            if filt:
                docs = filt

        # Trim to top_k after filtering
        docs = docs[: body.top_k]

        ctx = [d["text"] for d in docs]
        extracted = extract_driver(ctx) if "revenue" in lower_q else None
        answer = extracted if extracted else generate_answer(ctx, body.text)
        return {"answer": answer, "chunks": docs, "mode": "rag"}

    @app.get("/")
    def root():
        return {"msg": "IntelSent++ ready. Use /query, /driver or /docs."}

    return app


app = create_app()
