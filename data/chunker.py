# Chunk raw 10-Ks into ~256-token windows.
# Reads the thin catalog: /app/data/sec_catalog/sec_docs.csv
# For each row (company, year, doc_id), reads /app/data/sec-edgar-filings/{company}/{year}/primary.txt
# Writes a single CSV of chunks to --out (file or directory)/sec_chunks.csv

from __future__ import annotations
import argparse, csv, os, re, sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "sec-edgar-filings"
CATALOG_CSV = BASE / "data" / "sec_catalog" / "sec_docs.csv"

CHUNK_SIZE = 256
STRIDE = 40
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def _clean_text_from_html(html: str) -> str:
    # Fallback; we usually read primary.txt, not HTML
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)

def _read_primary_text(company: str, year: int) -> str:
    p_txt = RAW_DIR / company / str(year) / "primary.txt"
    if p_txt.exists():
        return p_txt.read_text(encoding="utf-8", errors="ignore")
    # fallback to html if needed
    p_html = RAW_DIR / company / str(year) / "primary.html"
    if p_html.exists():
        return _clean_text_from_html(p_html.read_text(encoding="utf-8", errors="ignore"))
    raise FileNotFoundError(f"Missing primary.{txt|html} for {company}/{year} under {RAW_DIR}")

def chunk_text(text: str, chunk_tokens: int = CHUNK_SIZE, stride: int = STRIDE) -> List[str]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[str] = []
    i, n = 0, len(ids)
    while i < n:
        window = ids[i:i+chunk_tokens]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if i + chunk_tokens >= n:
            break
        i += max(1, chunk_tokens - stride)
    return chunks

def resolve_out_path(out_arg: str) -> Path:
    p = Path(out_arg)
    if p.exists() and p.is_dir():
        return p / "sec_chunks.csv"
    if str(p).endswith(".csv"):
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    # treat as directory
    p.mkdir(parents=True, exist_ok=True)
    return p / "sec_chunks.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(CATALOG_CSV), help="CSV with columns: doc_id,company,year,...")
    ap.add_argument("--input-root", default=str(RAW_DIR), help="Root dir of raw filings (primary.txt present)")
    ap.add_argument("--out", required=True, help="Output CSV file or directory for sec_chunks.csv")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    if not catalog.exists():
        raise FileNotFoundError(f"{catalog} not found. Run data/fetch_edgar_api.py first.")
    out_csv = resolve_out_path(args.out)

    df = pd.read_csv(catalog)
    req = {"doc_id", "company", "year"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{catalog} missing columns: {miss}")

    rows: List[Dict[str, Any]] = []
    chunk_id = 0

    print(f"Chunking with ~{CHUNK_SIZE} tokens, stride {STRIDE}")
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        company = str(r["company"]).strip()
        year = int(r["year"])
        try:
            text = _read_primary_text(company, year)
            parts = chunk_text(text)
            for p in parts:
                rows.append({
                    "chunk_id": chunk_id,
                    "doc_id": str(r["doc_id"]),
                    "company": company,
                    "year": year,
                    "text": p,
                })
                chunk_id += 1
        except Exception as e:
            print(f"[skip] {company} {year}: {e}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chunk_id", "doc_id", "company", "year", "text"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} chunks -> {out_csv}")

if __name__ == "__main__":
    main()
