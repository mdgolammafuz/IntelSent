# Fetch 10-K primary HTML for given companies & years via SEC Submissions API.
# Writes: datasets/sec/sec_docs.csv with columns:
# [doc_id, company, year, primary_url, text_chars, primary_doc]
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
DATA_DIR = BASE / "datasets" / "sec"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA_DIR / "sec_docs.csv"

USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "IntelSent Research (Md Golam Mafuz <golammafuzgm@gmail.com>)"
)

HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles
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
    # Lightweight map (SEC also publishes a full list; this path avoids extra calls)
    # If not found here, try the full SEC ticker map endpoint.
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        # data is { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
        for _, v in data.items():
            if v.get("ticker", "").upper() == ticker.upper():
                return str(v["cik_str"]).zfill(10)
    except Exception:
        pass
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
        # year match
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
    # pick the most recent in that year
    items.sort(key=lambda x: x["filingDate"], reverse=True)
    return items[0]

def _primary_url(cik: str, accession: str, primary: str) -> str:
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary}"

def fetch_10k_html_for(ticker: str, year: int) -> Optional[Dict[str, Any]]:
    cik = _lookup_cik_by_ticker(ticker)
    if not cik:
        print(f"[skip] {ticker} {year}: CIK not found")
        return None
    subm = _get_company_submissions(cik)
    item = _pick_10k_accession(subm, year)
    if not item:
        print(f"[skip] {ticker} {year}: 10-K not found in submissions")
        return None
    url = _primary_url(cik, item["accession"], item["primary"])
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    html = r.text
    text = _clean_text(html)
    return {
        "doc_id": item["accession"],
        "company": ticker.upper(),
        "year": year,
        "primary_url": url,
        "text_chars": len(text),
        "primary_doc": item["primary"],
        "html": html,  # not written to CSV; used by chunker later if you want
    }

def _write_csv(rows: List[Dict[str, Any]]):
    fieldnames = ["doc_id", "company", "year", "primary_url", "text_chars", "primary_doc"]
    exist = OUT_CSV.exists()
    seen = set()
    if exist:
        csv.field_size_limit(1_000_000_000)
        # build seen keys to avoid duplicates
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
    ap.add_argument("--companies", nargs="+", default=["AAPL", "MSFT"], help="Tickers")
    ap.add_argument("--years", nargs="+", type=int, default=[2022], help="Filing years")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds between SEC requests")
    args = ap.parse_args()

    print(f"User-Agent: {USER_AGENT}")
    all_rows = []
    for t in args.companies:
        for y in args.years:
            try:
                rec = fetch_10k_html_for(t, y)
                if rec:
                    print(f"{t} {y} {rec['doc_id']} ({rec['text_chars']} chars) primary={rec['primary_doc']}")
                    all_rows.append(rec)
                time.sleep(args.sleep)
            except requests.HTTPError as e:
                print(f"[skip] {t} {y}: HTTPError {e}")
            except Exception as e:
                print(f"[skip] {t} {y}: {e}")
    if all_rows:
        _write_csv(all_rows)
    else:
        print("No new docs.")

if __name__ == "__main__":
    main()
