from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.generator import generate_answer
from rag.selector import extract_candidates, load_selector
from utils.canon import canonicalize

# ---------- App ----------
app = FastAPI(title="IntelSent SEC RAG", version="0.1.0")

# Lazy-initialized singletons
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[CrossReranker] = None
_selector = None  # learned selector (optional)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
SELECTOR_PATH = os.path.join(ARTIFACTS_DIR, "selector.pkl")


# ---------- Models ----------
class QueryRequest(BaseModel):
    text: str
    use_rerank: bool = True
    top_k: int = 5
    company: Optional[str] = None
    group_hint: Optional[str] = None  # "cloud"|"office"|"search"|"xbox"|"windows"|"iphone"


class QueryResponse(BaseModel):
    answer: str
    chunks: List[Dict[str, Any]]
    mode: str


class DriverRequest(BaseModel):
    company: str
    year: Optional[int] = None  # reserved for multi-year in future


class DriversBySegmentRequest(BaseModel):
    company: str
    top_k: int = 5  # how many chunks to scan
    year: Optional[int] = None  # reserved


class DriversBySegmentResponse(BaseModel):
    company: str
    segments: Dict[str, Dict[str, Any]]  # segment -> {driver, chunk}
    notes: Optional[str] = None


# ---------- Startup ----------
@app.on_event("startup")
def startup():
    global _retriever, _reranker, _selector  # local module-level, not Python "global" state elsewhere
    if _retriever is None:
        # use same knobs you’ve been using during eval
        _retriever = HybridRetriever(alpha=0.85, top_k=10000, candidate_pool=10000, bm25_pool=10000)
    if _reranker is None:
        _reranker = CrossReranker()
    if _selector is None:
        _selector = load_selector(SELECTOR_PATH)


# ---------- Helpers ----------
def _company_filter(docs: List[Dict[str, Any]], company: Optional[str]) -> List[Dict[str, Any]]:
    if not company:
        return docs
    filtered = [d for d in docs if d.get("company") == company]
    return filtered or docs

def _prioritize_by_category(docs: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    import re
    REV = re.compile(r"\b(revenue|sales)\b", re.I)
    MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
    OPEX = re.compile(r"\b(operating expenses|opex|research and development|sales and marketing|general and administrative)\b", re.I)
    DRV = re.compile(r"(driven by|led by|primarily by|resulting from|due to|because of)", re.I)

    def is_cat(d):
        t = d["text"].lower()
        if category == "revenue_driver":
            return bool(REV.search(t) and DRV.search(t))
        if category == "margin_driver":
            return bool(MARGIN.search(t) and DRV.search(t))
        if category == "opex_driver":
            return bool(OPEX.search(t) and DRV.search(t))
        return False

    driver_docs = [d for d in docs if is_cat(d)]
    seen = set()
    out: List[Dict[str, Any]] = []
    for d in driver_docs + docs:
        key = (d["doc_id"], d["chunk_id"])
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out

def _answer_with_selector(question: str, docs: List[Dict[str, Any]], category: Optional[str], group_hint: Optional[str]) -> Optional[str]:
    # Try learned selector first
    cands = extract_candidates(docs, category)
    if cands:
        pred = _selector.select(cands, category, group_hint=group_hint)
        if pred:
            return pred
    # Fallback: rule extractor → generator
    ctx = [d["text"] for d in docs]
    return extract_driver(ctx, category=category) or generate_answer(ctx, question) or ""

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "service": "IntelSent SEC RAG"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = req.text
    hits = _retriever.retrieve(q)
    docs = _retriever.docs_from_hits(hits)

    if req.use_rerank:
        docs = _reranker.rerank(q, docs, top_k=min(400, len(docs)))

    docs = _company_filter(docs, req.company)
    # generic Q&A path uses no strict category; but we still prefer revenue-style chunks first
    docs = _prioritize_by_category(docs, category="revenue_driver")[: max(1, req.top_k)]

    # try selector with optional group hint
    ans = _answer_with_selector(q, docs, category="revenue_driver", group_hint=req.group_hint)
    return QueryResponse(answer=ans, chunks=[{k: d[k] for k in d if k != "vector"} for d in docs], mode="rag")

@app.post("/driver", response_model=QueryResponse)
def driver(req: DriverRequest):
    # Deterministic scan for the “headline” driver (fast path)
    q = "Which product or service was revenue growth driven by?"
    hits = _retriever.retrieve(q)
    docs = _retriever.docs_from_hits(hits)
    docs = _reranker.rerank(q, docs, top_k=min(400, len(docs)))
    docs = _company_filter(docs, req.company)
    # hard-prioritize driver-like sentences, then take one
    docs = _prioritize_by_category(docs, category="revenue_driver")[:1]
    if not docs:
        return QueryResponse(answer="not found in context", chunks=[], mode="deterministic-scan")
    # extract short phrase from the top chunk only
    ans = _answer_with_selector(q, docs, category="revenue_driver", group_hint=None)
    return QueryResponse(answer=ans or "not found in context", chunks=[{k: docs[0][k] for k in docs[0] if k != "vector"}], mode="deterministic-scan")

@app.post("/drivers_by_segment", response_model=DriversBySegmentResponse)
def drivers_by_segment(req: DriversBySegmentRequest):
    """
    Return short 'drivers' per major segment for a company (e.g., cloud/office/search/xbox/windows/iphone),
    with the supporting chunk as citation.
    """
    q = "Which product or service was revenue growth driven by?"
    hits = _retriever.retrieve(q)
    docs = _retriever.docs_from_hits(hits)
    docs = _reranker.rerank(q, docs, top_k=min(600, len(docs)))
    docs = _company_filter(docs, req.company)
    docs = _prioritize_by_category(docs, category="revenue_driver")[: max(5, req.top_k)]

    # Define segments and hints
    segments = ["cloud", "office", "search", "xbox", "windows", "iphone"]
    out: Dict[str, Dict[str, Any]] = {}

    for seg in segments:
        # filter candidates to those containing the seg term to improve precision
        seg_docs = []
        for d in docs:
            t = d["text"].lower()
            if seg == "cloud" and ("azure" in t or "cloud services" in t or "intelligent cloud" in t):
                seg_docs.append(d)
            elif seg == "office" and ("office 365" in t or "microsoft 365" in t or "office commercial" in t):
                seg_docs.append(d)
            elif seg == "search" and ("search" in t or "news advertising" in t):
                seg_docs.append(d)
            elif seg == "xbox" and ("xbox" in t or "game pass" in t):
                seg_docs.append(d)
            elif seg == "windows" and ("windows oem" in t or "windows commercial" in t or "windows revenue" in t):
                seg_docs.append(d)
            elif seg == "iphone" and ("iphone" in t):
                seg_docs.append(d)

        subset = seg_docs or docs
        ans = _answer_with_selector(q, subset, category="revenue_driver", group_hint=seg)
        if ans:
            out[seg] = {
                "driver": canonicalize(ans),
                "chunk": {k: subset[0][k] for k in subset[0] if k != "vector"} if subset else None,
            }

    return DriversBySegmentResponse(company=req.company, segments=out, notes="Drivers are short phrases; see chunk for citation.")
