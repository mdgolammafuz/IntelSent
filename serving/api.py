
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import yaml

# local modules
from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.generator import generate_answer

# ----- Load config -----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "drivers.yml")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG: Dict[str, Any] = {}

# ----- FastAPI -----
app = FastAPI(title="IntelSent API", version="0.2")

class QueryRequest(BaseModel):
    text: str
    company: Optional[str] = None
    year: Optional[int] = None
    top_k: int = 5
    use_rerank: bool = True
    group_hint: Optional[str] = None  # e.g., "cloud", "search", etc.

class DriverRequest(BaseModel):
    company: str
    year: Optional[int] = None
    top_k: int = 8

class IngestRequest(BaseModel):
    companies: List[str]
    years: List[int]

def _retriever() -> HybridRetriever:
    # light defaults
    return HybridRetriever(alpha=0.85, top_k=50, candidate_pool=400, bm25_pool=800)

def _reranker() -> CrossReranker:
    return CrossReranker(batch_size=32, max_pairs=400)

def _filter_company_year(docs: List[Dict[str, Any]], company: Optional[str], year: Optional[int]) -> List[Dict[str, Any]]:
    if company:
        docs = [d for d in docs if str(d.get("company", "")).upper() == company.upper()] or docs
    if year:
        docs = [d for d in docs if int(d.get("year", 0)) == int(year)] or docs
    return docs

@app.on_event("startup")
def startup() -> None:
    global CONFIG
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    CONFIG = load_config(CONFIG_PATH)

@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "IntelSent", "config_loaded": bool(CONFIG)}

@app.post("/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    retr = _retriever()
    rr = _reranker()

    hits = retr.retrieve(req.text)
    docs = retr.docs_from_hits(hits)
    docs = _filter_company_year(docs, req.company, req.year)

    if req.use_rerank and docs:
        docs = rr.rerank(req.text, docs, top_k=min(req.top_k * 3, len(docs)))

    # Prepare contexts
    ctx = [d["text"] for d in docs[: req.top_k]]

    # Extraction patterns from config
    patterns = CONFIG.get("driver_regex", []) or []

    # If the question likely asks a 'driver', try extractor first
    answer = extract_driver(ctx, patterns)
    mode = "rag"
    if not answer:
        # fall back to small generator for a concise phrase
        answer = generate_answer(ctx, req.text) or "not found in context"

    # Attach top chunks for citation
    out_chunks = []
    for d in docs[: req.top_k]:
        out_chunks.append({
            "chunk_id": d.get("chunk_id"),
            "doc_id": d.get("doc_id"),
            "company": d.get("company"),
            "year": d.get("year"),
            "score": d.get("score", None),
            "text": d.get("text"),
            "ce_score": d.get("ce_score", None),
        })

    return {
        "answer": answer,
        "chunks": out_chunks,
        "mode": mode,
    }

@app.post("/driver")
def driver(req: DriverRequest) -> Dict[str, Any]:
    """
    Convenience route: ask the canonical 'revenue_driver' question from config for a given company/year.
    """
    q = CONFIG.get("questions", {}).get("revenue_driver", "Which product or service was revenue growth driven by?")
    q_req = QueryRequest(text=q, company=req.company, year=req.year, top_k=req.top_k, use_rerank=True)
    return query(q_req)

@app.post("/drivers_by_segment")
def drivers_by_segment(req: DriverRequest) -> Dict[str, Any]:
    """
    For each configured segment, try to extract a short driver phrase and cite the best chunk.
    """
    segments: Dict[str, List[str]] = CONFIG.get("segments", {}) or {}
    patterns = CONFIG.get("driver_regex", []) or []
    retr = _retriever()
    rr = _reranker()

    out: Dict[str, Any] = {"company": req.company, "segments": {}}

    # For each segment, use the canonical revenue question, then prefer chunks matching any segment keywords
    base_q = CONFIG.get("questions", {}).get("revenue_driver", "Which product or service was revenue growth driven by?")

    for seg_name, keywords in segments.items():
        hits = retr.retrieve(base_q)
        docs = retr.docs_from_hits(hits)
        docs = _filter_company_year(docs, req.company, req.year)
        if not docs:
            out["segments"][seg_name] = {"driver": None, "chunk": None}
            continue

        docs = rr.rerank(base_q, docs, top_k=min(max(8, req.top_k), len(docs)))

        # Prefer a doc containing any keyword (case-insensitive)
        pick = None
        kl = [k.lower() for k in keywords]
        for d in docs:
            t = (d.get("text") or "").lower()
            if any(k in t for k in kl):
                pick = d
                break
        if not pick:
            pick = docs[0]

        phrase = extract_driver([pick["text"]], patterns) or ""
        # If extractor missed, fall back to the first keyword found in the picked chunk
        if not phrase:
            low = (pick.get("text") or "").lower()
            for k in kl:
                if k in low:
                    phrase = k
                    break
        # If still empty, use a minimal generator pass on top-3 docs for that segment
        if not phrase:
            phrase = generate_answer([d["text"] for d in docs[:3]], base_q)

        out["segments"][seg_name] = {
            "driver": phrase,
            "chunk": {
                "chunk_id": pick.get("chunk_id"),
                "doc_id": pick.get("doc_id"),
                "company": pick.get("company"),
                "year": pick.get("year"),
                "score": pick.get("score", None),
                "text": pick.get("text"),
                "ce_score": pick.get("ce_score", None),
            },
        }

    out["notes"] = "Drivers are short phrases; see chunk for citation."
    return out

@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """
    Optional convenience to fetch->chunk->embed within one call.
    """
    import subprocess, shlex

    comps = req.companies
    years = [str(y) for y in req.years]

    cmds = [
        f'python {os.path.join(BASE_DIR, "data", "fetch_edgar_api.py")} --companies {" ".join(comps)} --years {" ".join(years)}',
        f'python {os.path.join(BASE_DIR, "data", "chunker.py")}',
        f'python {os.path.join(BASE_DIR, "data", "embedder.py")}',
    ]

    steps = []
    for c in cmds:
        p = subprocess.run(shlex.split(c), capture_output=True, text=True)
        steps.append({
            "cmd": c,
            "returncode": p.returncode,
            "stdout": p.stdout[-4000:],
            "stderr": p.stderr[-2000:],
        })
        if p.returncode != 0:
            break

    return {"steps": steps}
