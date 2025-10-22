# serving/api.py  (Pydantic v2-safe, no forward-refs)

import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

# make "rag/..." imports work when running inside container
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends, Body
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
import yaml

from serving.logging_setup import logger as log
from serving.settings import Settings
from serving.cache import get_cache, make_key

from rag.chain import load_chain
from rag.extract import extract_driver
from rag.generator import generate_answer

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# --------------------------------------------------------------------------------------
# Config / settings
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVERS_CFG_PATH = os.path.join(BASE_DIR, "config", "drivers.yml")

settings = Settings()

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# --------------------------------------------------------------------------------------
# Schemas (define BEFORE routes; no forward refs)
# --------------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    text: str
    company: Optional[str] = None
    year: Optional[int] = None
    top_k: int = 5
    no_openai: bool = False  # if True, force local path

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
# Chain / cache / auth / limiter
# --------------------------------------------------------------------------------------
SKIP_CHAIN_INIT = os.getenv("SKIP_CHAIN_INIT", "").lower() in {"1", "true", "yes"}
CHAIN = None if SKIP_CHAIN_INIT else load_chain(settings.app_cfg_path)

CACHE = get_cache(ttl_s=int(os.getenv("CACHE_TTL_SECONDS", "600")))

API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}

def require_api_key(x_api_key: Optional[str] = Header(None)):
    if not API_KEYS:  # dev mode if no keys configured
        return
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)

# LangSmith (no-op if disabled)
_LANGSMITH_ENABLED = bool(settings.langsmith_tracing)
try:
    if _LANGSMITH_ENABLED:
        from langsmith import traceable  # type: ignore
        log.info("langsmith.init", project=settings.langsmith_project)
    else:
        def traceable(*args, **kwargs):  # type: ignore
            def _wrap(fn): return fn
            return _wrap
except Exception as e:
    log.warning("langsmith.init_failed", error=str(e))
    def traceable(*args, **kwargs):  # type: ignore
        def _wrap(fn): return fn
        return _wrap

@traceable(name="query", run_type="chain")
def traced_chain_run(
    question: str,
    company: Optional[str],
    year: Optional[int],
    top_k: int,
    use_openai: bool,
) -> Dict[str, Any]:
    CHAIN.use_openai = use_openai
    return CHAIN.run(
        question=question,
        company=company,
        year=year,
        top_k=top_k,
    )

# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
app = FastAPI(title="IntelSent API", version="0.5")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Access log
@app.middleware("http")
async def access_log_mw(request: Request, call_next):
    rid = str(uuid.uuid4())
    start = time.perf_counter()
    request.state.request_id = rid
    try:
        response: Response = await call_next(request)
        dur_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query or ""),
            status=response.status_code,
            duration_ms=round(dur_ms, 2),
            request_id=rid,
        )
        response.headers["X-Request-ID"] = rid
        return response
    except Exception as e:
        dur_ms = (time.perf_counter() - start) * 1000.0
        log.error(
            "http.error",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query or ""),
            error=str(e),
            duration_ms=round(dur_ms, 2),
            request_id=rid,
        )
        raise

@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)

# Ingest router (optional)
try:
    from serving.ingest_runner import router as ingest_router
    app.include_router(ingest_router)
    log.info("router.ingest_loaded", ok=True)
except Exception as e:
    log.info("router.ingest_loaded", ok=False)
    log.warning("ingest.router_not_loaded", error=str(e))

# --------------------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------------------
DRIVERS_CFG: Dict[str, Any] = {}

@app.on_event("startup")
def startup() -> None:
    global DRIVERS_CFG
    DRIVERS_CFG = _load_yaml(DRIVERS_CFG_PATH)
    log.info(
        "app.startup",
        app_config=settings.app_cfg_path,
        drivers_cfg_loaded=bool(DRIVERS_CFG),
        chain_initialized=CHAIN is not None,
        config=settings.redacted(),
    )

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "IntelSent",
        "config": settings.redacted(),
        "drivers_cfg_loaded": bool(DRIVERS_CFG),
        "chain_initialized": CHAIN is not None,
    }

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "# intelsent demo metrics\nok 1\n"

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute;2/second")
def query(
    request: Request,
    req: QueryRequest = Body(...),
    _=Depends(require_api_key),
) -> QueryResponse:
    if CHAIN is None:
        log.warning("chain.not_initialized", route="/query")
        raise HTTPException(status_code=503, detail="CHAIN not initialized")

    log.info(
        "query.start",
        company=req.company,
        year=req.year,
        top_k=req.top_k,
        use_openai=not req.no_openai,
    )

    cache_key = make_key("q:v1", {
        "text": req.text,
        "company": req.company,
        "year": req.year,
        "top_k": req.top_k,
        "use_openai": not req.no_openai,
    })
    cached = CACHE.get(cache_key)
    if cached:
        log.info("cache.hit", route="/query")
        return QueryResponse(**cached)

    out = traced_chain_run(
        question=req.text,
        company=req.company,
        year=req.year,
        top_k=req.top_k,
        use_openai=not req.no_openai,
    )

    CACHE.set(cache_key, out)
    log.info("query.done", contexts=len(out.get("contexts", []) if out else []))
    return QueryResponse(**out)

@app.post("/driver", response_model=QueryResponse)
@limiter.limit("10/minute")
def driver(
    request: Request,
    req: DriverRequest = Body(...),
    _=Depends(require_api_key),
) -> QueryResponse:
    if CHAIN is None:
        log.warning("chain.not_initialized", route="/driver")
        raise HTTPException(status_code=503, detail="CHAIN not initialized")

    q_default = "Which product or service was revenue growth driven by?"
    q = (DRIVERS_CFG.get("questions", {}) or {}).get("revenue_driver", q_default)

    cache_key = make_key("driver:v1", {
        "q": q,
        "company": req.company,
        "year": req.year,
        "top_k": req.top_k,
        "use_openai": not req.no_openai,
    })
    cached = CACHE.get(cache_key)
    if cached:
        log.info("cache.hit", route="/driver")
        return QueryResponse(**cached)

    out = traced_chain_run(
        question=q,
        company=req.company,
        year=req.year,
        top_k=req.top_k,
        use_openai=not req.no_openai,
    )

    CACHE.set(cache_key, out)
    return QueryResponse(**out)

@app.post("/drivers_by_segment")
@limiter.limit("5/minute")
def drivers_by_segment(
    request: Request,
    req: DriverRequest = Body(...),
    _=Depends(require_api_key),
) -> Dict[str, Any]:
    if CHAIN is None:
        log.warning("chain.not_initialized", route="/drivers_by_segment")
        raise HTTPException(status_code=503, detail="CHAIN not initialized")

    segments: Dict[str, List[str]] = (DRIVERS_CFG.get("segments", {}) or {})
    patterns = (DRIVERS_CFG.get("driver_regex", []) or [])
    q_default = "Which product or service was revenue growth driven by?"
    base_q = (DRIVERS_CFG.get("questions", {}) or {}).get("revenue_driver", q_default)

    out: Dict[str, Any] = {"company": req.company, "segments": {}}
    for seg_name, keywords in segments.items():
        result = traced_chain_run(
            question=base_q,
            company=req.company,
            year=req.year,
            top_k=max(8, req.top_k),
            use_openai=not req.no_openai,
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
