import os, sys, time, uuid
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends, Body, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVERS_CFG_PATH = os.path.join(BASE_DIR, "config", "drivers.yml")
settings = Settings()

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class QueryRequest(BaseModel):
    text: str
    company: Optional[str] = None
    year: Optional[int] = None
    top_k: int = 5
    no_openai: bool = False

class QueryResponse(BaseModel):
    answer: str
    contexts: List[str]
    meta: Dict[str, Any]

class DriverRequest(BaseModel):
    company: str
    year: Optional[int] = None
    top_k: int = 8
    no_openai: bool = False

SKIP_CHAIN_INIT = os.getenv("SKIP_CHAIN_INIT", "").lower() in {"1","true","yes"}
CHAIN = None if SKIP_CHAIN_INIT else load_chain(settings.app_cfg_path)

CACHE = get_cache(ttl_s=int(os.getenv("CACHE_TTL_SECONDS", "600")))
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}

def _check_key(x_api_key: Optional[str], request: Request) -> None:
    if not API_KEYS:
        return
    qp_key = request.query_params.get("api_key")
    key = x_api_key or qp_key
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def require_api_key(request: Request, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key, request)

limiter = Limiter(key_func=get_remote_address, storage_uri=os.getenv("REDIS_URL","memory://"))

# LangSmith safe no-op
try:
    from langsmith import traceable  # type: ignore
except Exception:
    def traceable(*args, **kwargs):
        def _wrap(fn): return fn
        return _wrap

@traceable(name="query", run_type="chain")
def traced_chain_run(question: str, company: Optional[str], year: Optional[int], top_k: int, use_openai: bool) -> Dict[str, Any]:
    CHAIN.use_openai = use_openai
    return CHAIN.run(question=question, company=company, year=year, top_k=top_k)

app = FastAPI(title="IntelSent API", version="0.5")

# CORS FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://intel-ui.vercel.app"],
    allow_origin_regex="https://.*\\.vercel\\.app",
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# wildcard OPTIONS to neutralize 405 from proxies
@app.options("/{rest_of_path:path}")
def _opt_any(rest_of_path: str) -> Response:
    return Response(status_code=204)

@app.middleware("http")
async def access_log_mw(request: Request, call_next):
    rid = str(uuid.uuid4()); start = time.perf_counter()
    request.state.request_id = rid
    try:
        resp: Response = await call_next(request)
        dur = (time.perf_counter()-start)*1000
        resp.headers["X-Request-ID"] = rid
        return resp
    except Exception:
        raise

@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse({"detail":"rate limit exceeded"}, status_code=429)

try:
    from serving.ingest_runner import router as ingest_router
    app.include_router(ingest_router)
except Exception:
    pass

DRIVERS_CFG: Dict[str, Any] = {}

@app.on_event("startup")
def startup() -> None:
    global DRIVERS_CFG
    DRIVERS_CFG = _load_yaml(DRIVERS_CFG_PATH)

@app.get("/")
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "IntelSent", "config": settings.redacted(), "drivers_cfg_loaded": bool(DRIVERS_CFG), "chain_initialized": CHAIN is not None}

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "# intelsent demo metrics\nok 1\n"

# ----- POST (kept for API clients) -----
@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute;2/second")
def query(request: Request, req: QueryRequest = Body(...), _=Depends(require_api_key)) -> QueryResponse:
    if CHAIN is None: raise HTTPException(status_code=503, detail="CHAIN not initialized")
    cache_key = make_key("q:v1", {"text":req.text,"company":req.company,"year":req.year,"top_k":req.top_k,"use_openai":not req.no_openai})
    cached = CACHE.get(cache_key)
    if cached: return QueryResponse(**cached)
    out = traced_chain_run(req.text, req.company, req.year, req.top_k, not req.no_openai)
    CACHE.set(cache_key, out)
    return QueryResponse(**out)

# ----- NEW: GET route (simple request; no preflight) -----
@app.get("/query_min", response_model=QueryResponse)
@limiter.limit("10/minute;2/second")
def query_min(
    request: Request,
    text: str = Query(...),
    company: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    top_k: int = Query(3),
    no_openai: bool = Query(True),
    api_key: Optional[str] = Query(None),  # picked up by _check_key via query_params
):
    _check_key(None, request)
    if CHAIN is None: raise HTTPException(status_code=503, detail="CHAIN not initialized")
    cache_key = make_key("qmin:v1", {"text":text,"company":company,"year":year,"top_k":top_k,"use_openai":not no_openai})
    cached = CACHE.get(cache_key)
    if cached: return QueryResponse(**cached)
    out = traced_chain_run(text, company, year, top_k, not no_openai)
    CACHE.set(cache_key, out)
    return QueryResponse(**out)
