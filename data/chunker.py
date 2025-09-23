
# Read datasets/sec/sec_docs.csv (doc_id, company, year, primary_url, text_chars, primary_doc)
# Download each primary_url (cached locally), clean to text, chunk into ~256-token windows,
# and write datasets/sec_chunks.csv with: chunk_id, doc_id, company, year, text
from __future__ import annotations

import os
import csv
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from tqdm import tqdm

BASE = Path(__file__).resolve().parents[1]
DOCS_CSV = BASE / "datasets" / "sec" / "sec_docs.csv"
OUT_CSV = BASE / "datasets" / "sec_chunks.csv"
CACHE_DIR = BASE / "datasets" / "sec" / "html"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "IntelSent Research (Md Golam Mafuz <golammafuzgm@gmail.com>)"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Chunking params
CHUNK_SIZE = 256
STRIDE = 40
TOKENIZER_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def fetch_html(doc_id: str, url: str, sleep: float = 0.8) -> str:
    """Fetch and cache the 10-K primary HTML by doc_id."""
    cache_path = CACHE_DIR / f"{doc_id}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    html = r.text
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(sleep)  # be polite to SEC
    return html

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text

def chunk_text(text: str, chunk_tokens: int = CHUNK_SIZE, stride: int = STRIDE) -> List[str]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[str] = []
    i = 0
    n = len(ids)
    while i < n:
        window = ids[i : i + chunk_tokens]
        if not window:
            break
        chunk = tokenizer.decode(window, skip_special_tokens=True)
        chunks.append(chunk)
        if i + chunk_tokens >= n:
            break
        i += max(1, chunk_tokens - stride)
    return chunks

def main():
    if not DOCS_CSV.exists():
        raise FileNotFoundError(f"{DOCS_CSV} not found. Run data/fetch_edgar_api.py first.")
    df = pd.read_csv(DOCS_CSV)
    # expected columns: doc_id, company, year, primary_url, text_chars, primary_doc
    required = {"doc_id", "company", "year", "primary_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{DOCS_CSV} missing columns: {missing}")

    rows: List[Dict[str, Any]] = []
    chunk_id = 0

    print(f"Token indices sequence length may exceed 512 in source, but we chunk to ~{CHUNK_SIZE} tokens.")

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        doc_id = str(r["doc_id"])
        company = str(r["company"])
        year = int(r["year"])
        url = str(r["primary_url"])

        try:
            html = fetch_html(doc_id, url)
            text = html_to_text(html)
            parts = chunk_text(text)
            for p in parts:
                rows.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "company": company,
                    "year": year,
                    "text": p,
                })
                chunk_id += 1
        except Exception as e:
            print(f"[skip] {company} {year} {doc_id}: {e}")

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chunk_id", "doc_id", "company", "year", "text"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Wrote {len(rows)} chunks to {OUT_CSV}")

if __name__ == "__main__":
    main()
