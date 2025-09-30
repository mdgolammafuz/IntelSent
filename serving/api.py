from __future__ import annotations

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml

# ---- Local modules ----
from rag.chain import load_chain
from rag.extract import extract_driver
from rag.generator import generate_answer

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVERS_CFG_PATH = os.path.join(BASE_DIR, "config", "drivers.yml")

# Prefer APP_CFG_PATH (compose), then APP_CONFIG, then default
APP_CFG_PATH = (
    os.getenv("APP_CFG_PATH")
    or os.getenv("APP_CONFIG")
    or os.path.join(BASE_DIR, "config", "app.yaml")
)

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

DRIVERS_CFG: Dict[str, Any] = {}

# --------------------------------------------------------------------------------------
# Chain init (test-friendly)
# --------------------------------------------------------------------------------------
SKIP_CHAIN_INIT = os.getenv("SKIP_CHAIN_INIT", "").lower() in {"1", "true", "yes"}
CHAIN = None if SKIP_CHAIN_INIT else load_chain(APP_CFG_PATH)

# --------------------------------------------------------------------------------------
# FastAPI app + optional ingest router
# --------------------------------------------------------------------------------------
app = FastAPI(title="IntelSent API", version="0.3")

try:
    from serving.ingest_runner import router as ingest_router
    app.include_router(ingest_router)
except Exception as e:
    print(f"[api] Ingest router not loaded: {e}")

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    text: str
    company: Optional[str] = None
    year: Optional[int] = None
    top_k: int = 5
    no_openai: bool = False  # force local answering if True

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    meta: Dict[str, Any]

class DriverRequest(BaseModel):
    company: str
    year: Optional[int] = None
    top_k: int = 8
    no_openai: bool = False

# --------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------
@app.on_event("startup")
def startup() -> None:
    global DRIVERS_CFG
    DRIVERS_CFG = _load_yaml(DRIVERS_CFG_PATH)

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "IntelSent",
        "app_config": APP_CFG_PATH,
        "drivers_cfg_loaded": bool(DRIVERS_CFG),
        "chain_initialized": CHAIN is not None,
    }

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if CHAIN is None:
        raise HTTPException(status_code=503, detail="CHAIN not initialized")
    CHAIN.use_openai = not req.no_openai
    out = CHAIN.run(
        question=req.text,
        company=req.company,
        year=req.year,
        top_k=req.top_k,
    )
    return QueryResponse(**out)

@app.post("/driver", response_model=QueryResponse)
def driver(req: DriverRequest) -> QueryResponse:
    if CHAIN is None:
        raise HTTPException(status_code=503, detail="CHAIN not initialized")
    q_default = "Which product or service was revenue growth driven by?"
    q = (DRIVERS_CFG.get("questions", {}) or {}).get("revenue_driver", q_default)
    CHAIN.use_openai = not req.no_openai
    out = CHAIN.run(
        question=q,
        company=req.company,
        year=req.year,
        top_k=req.top_k,
    )
    return QueryResponse(**out)

@app.post("/drivers_by_segment")
def drivers_by_segment(req: DriverRequest) -> Dict[str, Any]:
    if CHAIN is None:
        raise HTTPException(status_code=503, detail="CHAIN not initialized")
    segments: Dict[str, List[str]] = (DRIVERS_CFG.get("segments", {}) or {})
    patterns = (DRIVERS_CFG.get("driver_regex", []) or [])
    q_default = "Which product or service was revenue growth driven by?"
    base_q = (DRIVERS_CFG.get("questions", {}) or {}).get("revenue_driver", q_default)

    CHAIN.use_openai = not req.no_openai

    out: Dict[str, Any] = {"company": req.company, "segments": {}}
    for seg_name, keywords in segments.items():
        result = CHAIN.run(
            question=base_q,
            company=req.company,
            year=req.year,
            top_k=max(8, req.top_k),
        )
        ctxs: List[str] = result.get("contexts", []) or []
        meta = result.get("meta", {})

        pick_text = None
        kl = [k.lower() for k in keywords]
        for t in ctxs:
            low = (t or "").lower()
            if any(k in low for k in kl):
                pick_text = t
                break
        if not pick_text and ctxs:
            pick_text = ctxs[0]

        phrase = ""
        if pick_text:
            phrase = extract_driver([pick_text], patterns) or ""
            if not phrase:
                phrase = generate_answer(ctxs[:3], base_q) or ""

        out["segments"][seg_name] = {
            "driver": phrase or None,
            "context": pick_text,
            "meta": meta,
        }

    out["notes"] = "Drivers are short phrases; see 'context' for a citation."
    return out
