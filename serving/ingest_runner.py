# serving/ingest_runner.py
import os
import shlex
import subprocess
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class IngestSECRequest(BaseModel):
    companies: List[str]
    years: List[int]
    # if true, we run fetch -> chunk -> embed in one go
    rebuild: bool = True
    # optional: override SEC_USER_AGENT if needed
    sec_user_agent: Optional[str] = None

def _run(cmd: str, env: Optional[Dict[str, str]] = None) -> str:
    proc = subprocess.run(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env or os.environ.copy(),
        check=False,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Command failed: {cmd}\n{proc.stdout}")
    return proc.stdout

@router.post("/ingest/sec")
def ingest_sec(req: IngestSECRequest) -> Dict[str, Any]:
    env = os.environ.copy()
    if req.sec_user_agent:
        env["SEC_USER_AGENT"] = req.sec_user_agent

    if "SEC_USER_AGENT" not in env:
        # fall back to a descriptive default
        env["SEC_USER_AGENT"] = "IntelSent Research (please-set-SEC_USER_AGENT@example.com)"

    # 1) fetch 10-Ks via your existing script
    comps = " ".join(req.companies)
    yrs = " ".join(str(y) for y in req.years)
    out_fetch = _run(f"python data/fetch_edgar_api.py --companies {comps} --years {yrs}", env=env)

    # 2) chunk
    out_chunk = _run("python data/chunker.py", env=env)

    # 3) embed + rebuild FAISS
    out_embed = _run("python data/embedder.py", env=env)

    return {
        "status": "ok",
        "steps": {
            "fetch": out_fetch.strip()[-2000:],  # last part of logs
            "chunk": out_chunk.strip()[-2000:],
            "embed": out_embed.strip()[-2000:],
        }
    }
