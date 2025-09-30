from __future__ import annotations
import subprocess, os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/ingest", tags=["ingest"])

class IngestSECRequest(BaseModel):
    companies: List[str]
    years: List[int]
    rebuild: bool = True
    sec_user_agent: Optional[str] = None

@router.post("/sec")
def ingest_sec(body: IngestSECRequest):
    env = dict(os.environ)
    if body.sec_user_agent:
        env["SEC_USER_AGENT"] = body.sec_user_agent

    # ensure we run from /app so relative path exists in container
    cwd = "/app"
    cmd = [
        "python",
        "data/fetch_edgar_api.py",
        "--companies", *body.companies,
        "--years", *map(str, body.years),
    ]
    if body.rebuild:
        cmd.append("--rebuild")

    try:
        out = subprocess.check_output(cmd, cwd=cwd, env=env, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "output": out}
    except subprocess.CalledProcessError as e:
        return {"detail": f"Command failed: {' '.join(cmd)}\n{e.output}"}
