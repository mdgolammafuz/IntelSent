# flows/prefect_a2a.py
from prefect import flow, task
import os, sys, subprocess, requests

API = os.getenv("INTELSENT_API", "http://localhost:8000")
CONTAINER = os.getenv("INTELSENT_CONTAINER", "intelsent-api")

# 🔒 Require key from env; no default
KEY = os.getenv("INTELSENT_KEY")
if not KEY:
    raise SystemExit("INTELSENT_KEY is not set. Export INTELSENT_KEY before running.")
H = {"X-API-Key": KEY}

def _dex(cmd: str) -> str:
    return subprocess.check_output(
        ["docker", "exec", "-i", CONTAINER, "bash", "-lc", cmd],
        text=True,
        stderr=subprocess.STDOUT,
    )

@task(retries=2, retry_delay_seconds=3)
def ping() -> None:
    r = requests.get(f"{API}/healthz", timeout=10)
    r.raise_for_status()

@task(retries=2, retry_delay_seconds=10)
def ensure_ingest(company: str, year: int, ua: str = "you@example.com") -> None:
    # Try /actions/status (preferred)
    try:
        s = requests.get(
            f"{API}/actions/status",
            params={"company": company, "year": year, "api_key": KEY},
            timeout=10,
        )
        if s.status_code == 200 and s.json().get("ingested"):
            return
    except Exception:
        pass
    # Try /actions/ingest
    try:
        r = requests.post(
            f"{API}/actions/ingest",
            json={"company": company, "year": year, "rebuild": False, "sec_user_agent": ua},
            headers=H,
            timeout=600,
        )
        if r.status_code == 200:
            return
    except Exception:
        pass
    # Fallback to legacy /ingest/sec
    r = requests.post(
        f"{API}/ingest/sec",
        json={"companies": [company], "years": [year], "rebuild": False, "sec_user_agent": ua},
        timeout=600,
    )
    r.raise_for_status()

@task(retries=1, retry_delay_seconds=3)
def chunk() -> str:
    return _dex(
        "python -u data/chunker.py "
        "--catalog /app/data/sec_catalog/sec_docs.csv "
        "--input-root /app/data/sec-edgar-filings "
        "--out /app/data/chunks/sec_chunks.csv"
    )

@task(retries=1, retry_delay_seconds=3)
def embed() -> str:
    return _dex(
        "python -u data/embedder.py "
        "--chunks /app/data/chunks/sec_chunks.csv "
        "--db-dsn postgresql://intel:intel@intel_pgvector:5432/intelrag "
        "--table chunks --batch-size 128"
    )

@task(retries=2, retry_delay_seconds=5)
def ask(company: str, year: int, text: str) -> dict:
    r = requests.get(
        f"{API}/query_min",
        params={
            "text": text,
            "company": company,
            "year": year,
            "top_k": 3,
            "no_openai": True,
            "api_key": KEY,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

@flow(name="IntelSent-A2A")
def a2a(company: str = "AAPL", year: int = 2023,
        text: str = "Which products drove revenue growth?") -> dict:
    ping()
    ensure_ingest(company, year)
    print("• chunking…");  chunk()
    print("• embedding…"); embed()
    print("• querying…")
    return ask(company, year, text)

def main() -> int:
    company = os.getenv("A2A_COMPANY", "AAPL")
    year = int(os.getenv("A2A_YEAR", "2023"))
    question = os.getenv("A2A_Q", "Which products drove revenue growth?")
    print(f"▶ A2A: {company} {year}")
    out = a2a(company, year, question)
    ans = out.get("answer", "") if isinstance(out, dict) else ""
    ctxs = out.get("contexts") if isinstance(out, dict) else []
    print("\n=== ANSWER ===\n", ans or "—")
    if ctxs:
        print("\n=== CONTEXTS (first 2) ===")
        for c in ctxs[:2]:
            print("-", (c or "").replace("\n", " ")[:400])
    return 0

if __name__ == "__main__":
    sys.exit(main())
