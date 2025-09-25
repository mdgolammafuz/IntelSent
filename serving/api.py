from __future__ import annotations

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import yaml

# ---- Local modules (new chain) ----
# This is the pgvector-backed chain we wired during the CLI testing.
from rag.chain import load_chain

# (Optional) legacy helpers still used by the /drivers_by_segment route
# We can remove these later if you don’t use that route.
from rag.extract import extract_driver
from rag.generator import generate_answer

# --------------------------------------------------------------------------------------
# Config loading (drivers.yml is optional; app.yaml is used by the chain internally)
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVERS_CFG_PATH = os.path.join(BASE_DIR, "config", "drivers.yml")
APP_CFG_PATH = os.getenv("APP_CONFIG", os.path.join(BASE_DIR, "config", "app.yaml"))

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

DRIVERS_CFG: Dict[str, Any] = {}

# --------------------------------------------------------------------------------------
# Create chain once at startup
# --------------------------------------------------------------------------------------
CHAIN = load_chain(APP_CFG_PATH)

# --------------------------------------------------------------------------------------
# FastAPI app + optional ingest router
# --------------------------------------------------------------------------------------
app = FastAPI(title="IntelSent API", version="0.3")

# Keep our existing ingest router wiring
try:
    from serving.ingest_runner import router as ingest_router
    app.include_router(ingest_router)
except Exception as e:
    # Non-fatal; route won't be present if the module isn't available
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
    }

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """
    Primary query endpoint. Uses the pgvector-backed chain.
    Set no_openai=true to force the local answerer even if OPENAI_API_KEY is set.
    """
    # toggle OpenAI usage per-request if we want
    CHAIN.use_openai = not req.no_openai

    out = CHAIN.run(
        question=req.text,
        company=req.company,
        year=req.year,
        top_k=req.top_k,
    )
    # out has: {"answer": str, "contexts": List[str], "meta": {...}}
    return QueryResponse(**out)

@app.post("/driver", response_model=QueryResponse)
def driver(req: DriverRequest) -> QueryResponse:
    """
    Convenience route: uses a canonical 'revenue driver' question if present in drivers.yml
    """
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

# Optional: If we still want a segmented “drivers_by_segment” helper, keep this.
# It uses the chain to get top contexts and then tries a tiny extractor/generator.
@app.post("/drivers_by_segment")
def drivers_by_segment(req: DriverRequest) -> Dict[str, Any]:
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

        # prefer a context that contains a segment keyword
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
                # last fallback: tiny generator over the top-3 contexts
                phrase = generate_answer(ctxs[:3], base_q) or ""

        out["segments"][seg_name] = {
            "driver": phrase or None,
            "context": pick_text,
            "meta": meta,
        }

    out["notes"] = "Drivers are short phrases; see 'context' for a citation."
    return out
