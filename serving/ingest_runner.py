from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Where we record that a (company,year) has been ingested
_INGEST_STAMP_DIR = pathlib.Path("/app/data/.ingested")
_INGEST_STAMP_DIR.mkdir(parents=True, exist_ok=True)


def _norm_ticker(t: str) -> str:
    return (t or "").strip().upper().replace(" ", "")


def _stamp_path(company: str, year: int) -> pathlib.Path:
    return _INGEST_STAMP_DIR / f"{_norm_ticker(company)}_{int(year)}.done"


def is_ingested(company: str, year: int) -> bool:
    """Return True if we already stamped this (company,year)."""
    return _stamp_path(company, year).exists()


def _run_fetch(
    companies: List[str],
    years: List[int],
    *,
    rebuild: bool,  # accepted but NOT forwarded (script doesn’t support it)
    sec_user_agent: Optional[str],
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if sec_user_agent:
        env["SEC_USER_AGENT"] = sec_user_agent

    cwd = "/app"  # ensure relative path exists in container
    cmd = [
        "python",
        "data/fetch_edgar_api.py",
        "--companies",
        *[_norm_ticker(c) for c in companies],
        "--years",
        *[str(int(y)) for y in years],
    ]
    # NOTE: fetch_edgar_api.py does NOT support --rebuild. We emulate rebuild at stamp layer.

    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_ingested(
    company: str,
    year: int,
    *,
    rebuild: bool = False,
    sec_user_agent: Optional[str] = None,
) -> bool:
    """
    Idempotent. If (company,year) not present, fetches from EDGAR and stamps it.
    Returns True when data is confirmed present (stamp exists), False otherwise.
    """
    company = _norm_ticker(company)
    year = int(year)

    if rebuild:
        p = _stamp_path(company, year)
        if p.exists():
            p.unlink(missing_ok=True)
    elif is_ingested(company, year):
        return True

    proc = _run_fetch([company], [year], rebuild=rebuild, sec_user_agent=sec_user_agent)
    if proc.returncode == 0:
        _stamp_path(company, year).touch()
        return True
    return False


# ------------------- API Router -------------------

class IngestSECRequest(BaseModel):
    companies: List[str]
    years: List[int]
    rebuild: bool = False
    sec_user_agent: Optional[str] = None


@router.post("/sec")
def ingest_sec(body: IngestSECRequest):
    # Emulate rebuild by clearing stamps first (script itself has no --rebuild)
    if body.rebuild:
        for c in body.companies:
            for y in body.years:
                _stamp_path(c, y).unlink(missing_ok=True)

    proc = _run_fetch(
        body.companies,
        body.years,
        rebuild=body.rebuild,
        sec_user_agent=body.sec_user_agent,
    )

    if proc.returncode != 0:
        # Return structured HTTP error with stdout/stderr tails
        cmd_str = "python data/fetch_edgar_api.py --companies " + " ".join(
            _norm_ticker(c) for c in body.companies
        ) + " --years " + " ".join(str(int(y)) for y in body.years)
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "command": cmd_str,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
                "returncode": proc.returncode,
            },
        )

    # Stamp every (company,year) that was requested
    for c in body.companies:
        for y in body.years:
            _stamp_path(c, y).touch()

    return {
        "ok": True,
        "companies": [_norm_ticker(c) for c in body.companies],
        "years": [int(y) for y in body.years],
        "bytes_stdout": len(proc.stdout or ""),
        "bytes_stderr": len(proc.stderr or ""),
    }


@router.get("/status")
def ingest_status(
    company: str = Query(..., description="Ticker, e.g., AAPL"),
    year: int = Query(..., description="Year, e.g., 2023"),
):
    return {
        "company": _norm_ticker(company),
        "year": int(year),
        "ingested": is_ingested(company, year),
    }
