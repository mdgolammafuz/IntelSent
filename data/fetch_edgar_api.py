# Fetch 10-K primary HTML for given companies & years via SEC Submissions API.
# Writes raw artifacts under /app/data/sec-edgar-filings/<CIK or TICKER>/<year>:
#   meta.json, primary.html, primary.txt
# Also appends a thin CSV catalog at /app/data/sec_catalog/sec_docs.csv
from __future__ import annotations

import os
import csv
import time
import json
import re
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]

# ---- DO NOT use a top-level "datasets/" directory; it collides with HF 'datasets' ----
RAW_DIR = BASE / "data" / "sec-edgar-filings"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_DIR = BASE / "data" / "sec_catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = CATALOG_DIR / "sec_docs.csv"

USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "IntelSent Research (Md Golam Mafuz <golammafuzgm@gmail.com>)"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")  # requires lxml installed
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text

def _get_company_submissions(cik: str | int) -> Dict[str, Any]:
    cik_str = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def _lookup_cik_by_ticker(ticker: str) -> Optional[str]:
    # SEC endpoint with full ticker map
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    for _, v in data.items():
        if v.get("ticker", "").upper() == ticker.upper():
            return str(v["cik_str"]).zfill(10)
    return None

def _pick_10k_accession(subm_json: Dict[str, Any], year: int) -> Optional[Dict[str, Any]]:
    filings = subm_json.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    acc = filings.get("accessionNumber", [])
    prim = filings.get("primaryDocument", [])
    fdate = filings.get("filingDate", [])
    items = []
    for i, f in enumerate(forms):
        if f != "10-K":
            continue
        try:
            y = int(str(fdate[i])[:4])
        except Exception:
            continue
        if y != year:
            continue
        items.append({
            "accession": acc[i],
            "primary": prim[i],
            "filingDate": fdate[i]
        })
    if not items:
        return None
    items.sort(key=lambda x: x["filingDate"], reverse=True)
    return items[0]

def _primary_url(cik: str, accession: str, primary: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary}"

def _write_raw_bundle(base_dir: Path, html: str, meta: Dict[str, Any]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "primary.html").write_text(html, encoding="utf-8")
    (base_dir / "primary.txt").write_text(_clean_text(html), encoding="utf-8")
    (base_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def fetch_10k_html_for(company: str, year: int) -> Optional[Dict[str, Any]]:
    """
    company may be a TICKER (AAPL) or a numeric CIK string.
    We normalize to CIK where possible for storage.
    """
    comp_norm = (company or "").strip().upper()
    if comp_norm.isdigit():
        cik = comp_norm.zfill(10)
        ticker = None
    else:
        ticker = comp_norm
        cik = _lookup_cik_by_ticker(ticker)
        if not cik:
            print(f"[skip] {company} {year}: CIK not found")
            return None

    subm = _get_company_submissions(cik)
    item = _pick_10k_accession(subm, year)
    if not item:
        print(f"[skip] {company} {year}: 10-K not found in submissions")
        return None

    url = _primary_url(cik, item["accession"], item["primary"])
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    html = r.text

    # Where to place raw files
    leaf = ticker if ticker else cik
    bundle_dir = RAW_DIR / leaf / str(year)
    meta = {
        "doc_id": item["accession"],
        "company": leaf,
        "cik": cik,
        "year": year,
        "primary_url": url,
        "primary_doc": item["primary"],
        "filing_date": item["filingDate"],
        "user_agent": USER_AGENT,
    }
    _write_raw_bundle(bundle_dir, html, meta)

    text_chars = len(_clean_text(html))
    return {
        "doc_id": item["accession"],
        "company": leaf,
        "year": year,
        "primary_url": url,
        "text_chars": text_chars,
        "primary_doc": item["primary"],
    }

def _write_csv(rows: List[Dict[str, Any]]):
    fieldnames = ["doc_id", "company", "year", "primary_url", "text_chars", "primary_doc"]
    exist = OUT_CSV.exists()
    seen = set()
    if exist:
        csv.field_size_limit(1_000_000_000)
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add((row["doc_id"], row["company"], int(row["year"])))
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exist:
            w.writeheader()
        wrote = 0
        for r in rows:
            key = (r["doc_id"], r["company"], int(r["year"]))
            if key in seen:
                continue
            w.writerow({k: r[k] for k in fieldnames})
            wrote += 1
        print(f"Wrote {wrote} docs to {OUT_CSV}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", nargs="+", default=["AAPL", "MSFT"], help="Tickers or CIKs")
    ap.add_argument("--years", nargs="+", type=int, default=[2023], help="Filing years")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds between SEC requests")
    args = ap.parse_args()

    print(f"User-Agent: {USER_AGENT}")
    all_rows = []
    for comp in args.companies:
        for y in args.years:
            try:
                rec = fetch_10k_html_for(comp, y)
                if rec:
                    print(f"[ok] {rec['company']} ({rec.get('cik','')}) {y} {rec['doc_id']} "
                          f"chars={rec['text_chars']} primary={rec['primary_doc']}")
                    all_rows.append(rec)
                time.sleep(args.sleep)
            except requests.HTTPError as e:
                print(f"[skip] {comp} {y}: HTTPError {e}")
            except Exception as e:
                print(f"[skip] {comp} {y}: {e}")
    if all_rows:
        _write_csv(all_rows)
        print(f"Fetched {len(all_rows)} docs.")
    else:
        print("No new docs.")

if __name__ == "__main__":
    main()
