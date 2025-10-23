from __future__ import annotations

import os
import glob
import pathlib
import subprocess
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Shared data locations (container path; host binds ./data -> /app/data)
_DATA_DIR = pathlib.Path("/app/data")
_RAW_FILINGS_DIR = _DATA_DIR / "sec-edgar-filings"
_INGEST_STAMP_DIR = _DATA_DIR / ".ingested"
_INGEST_STAMP_DIR.mkdir(parents=True, exist_ok=True)


def _norm_ticker(t: str) -> str:
    return (t or "").strip().upper().replace(" ", "")


def _stamp_path(company: str, year: int) -> pathlib.Path:
    return _INGEST_STAMP_DIR / f"{_norm_ticker(company)}_{int(year)}.done"


def is_ingested(company: str, year: int) -> bool:
    """Return True if we already stamped this (company,year)."""
    return _stamp_path(company, year).exists()


def _filings_glob(company: str, year: int) -> List[str]:
    """
    Return matching raw filing file paths for (company,year).
    Expected layout:
      /app/data/sec-edgar-filings/{COMPANY_OR_CIK}/{YEAR}/**/*
    """
    c = _norm_ticker(company)
    y = int(year)
    base = _RAW_FILINGS_DIR / c / str(y)
    return glob.glob(str(base / "**" / "*"), recursive=True)


def _downloaded_count(company: str, year: int) -> int:
    """Count regular files (not dirs) under the expected raw filings dir."""
    return sum(1 for p in _filings_glob(company, year) if os.path.isfile(p))


def _run_fetch(
    companies: List[str],
    years: List[int],
    *,
    rebuild: bool,  # accepted but not forwarded (script doesn’t support it)
    sec_user_agent: Optional[str],
) -> subprocess.CompletedProcess:
    """
    Invoke the fetcher script. We do NOT pass --rebuild because the script doesn’t support it.
    Rebuild is emulated at the stamp layer only.
    """
    env = dict(os.environ)
    if sec_user_agent:
        env["SEC_USER_AGENT"] = sec_user_agent

    cmd = [
        "python",
        "data/fetch_edgar_api.py",
        "--companies",
        *[_norm_ticker(c) for c in companies],
        "--years",
        *[str(int(y)) for y in years],
    ]

    return subprocess.run(
        cmd,
        cwd="/app",
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
    Returns True when data is confirmed present (files exist and stamp written), False otherwise.
    """
    company = _norm_ticker(company)
    year = int(year)

    # Emulate rebuild by clearing the stamp first (we don’t delete files on disk).
    if rebuild:
        _stamp_path(company, year).unlink(missing_ok=True)
    elif is_ingested(company, year):
        return True

    proc = _run_fetch([company], [year], rebuild=rebuild, sec_user_agent=sec_user_agent)
    if proc.returncode != 0:
        return False

    # Only stamp if files actually exist.
    if _downloaded_count(company, year) > 0:
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
    """
    Kick off EDGAR fetch for the given (company, year) pairs.
    - Emulates rebuild by clearing stamps.
    - Does NOT pass --rebuild to the script (unsupported).
    - Stamps only when files are present under /app/data/sec-edgar-filings/{company}/{year}.
    - Returns stdout/stderr for diagnostics plus per-pair file counts.
    """
    # Emulate rebuild: clear stamps so subsequent checks won’t short-circuit.
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

    stamped: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for c in body.companies:
        for y in body.years:
            cnt = _downloaded_count(c, y)
            entry = {"company": _norm_ticker(c), "year": int(y), "files": cnt}
            if cnt > 0:
                _stamp_path(c, y).touch()
                stamped.append(entry)
            else:
                missing.append(entry)

    resp = {
        "ok": proc.returncode == 0,
        "companies": [_norm_ticker(c) for c in body.companies],
        "years": [int(y) for y in body.years],
        "stamped": stamped,
        "missing": missing,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or ""),
        "stderr": (proc.stderr or ""),
    }

    # Surface command failure as 502 with diagnostics, do not stamp on failure.
    if proc.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail=resp,
        )

    return resp


@router.get("/status")
def ingest_status(
    company: str = Query(..., description="Ticker or CIK, e.g., AAPL or 0000320193"),
    year: int = Query(..., description="Year, e.g., 2023"),
):
    c = _norm_ticker(company)
    y = int(year)
    return {
        "company": c,
        "year": y,
        "ingested": is_ingested(c, y),
        "files_found": _downloaded_count(c, y),
        "raw_dir": str(_RAW_FILINGS_DIR / c / str(y)),
    }
