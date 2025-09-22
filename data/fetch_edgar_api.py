import os, csv, re, time, json, requests
from pathlib import Path
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "datasets" / "sec"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_CSV = DATA_DIR / "sec_docs.csv"

UA = os.environ.get(
    "SEC_USER_AGENT",
    "IntelSent Research (Your Name <you@example.com>)"
)
if "@" not in UA:
    raise SystemExit("Set SEC_USER_AGENT to a real contact string. Example:\n"
                     "export SEC_USER_AGENT=\"IntelSent Research (Md Golam Mafuz <you@email>)\"")

# Known CIKs to avoid a separate mapping call
COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
}
YEARS = {2022}  # adjust if you want more

S = requests.Session()
S.headers.update({"User-Agent": UA})

def get_submissions_json(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = S.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def build_archive_url(cik: str, accession_no: str, primary_doc: str) -> str:
    # accession no in JSON is like "0000320193-22-000108" → remove dashes for path
    acc_path = accession_no.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_path}/{primary_doc}"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_primary_doc(cik: str, accession_no: str, primary_doc: str) -> str:
    url = build_archive_url(cik, accession_no, primary_doc)
    r = S.get(url, timeout=120, headers={"Accept": "text/html"})
    r.raise_for_status()
    # Some docs are HTML, some are plain text
    content_type = r.headers.get("Content-Type", "").lower()
    if "html" in content_type or url.lower().endswith((".htm", ".html")):
        return html_to_text(r.text)
    else:
        # treat as plain text; still normalize spaces
        return re.sub(r"\s+", " ", r.text).strip()

def collect_10k_docs():
    out_rows = []
    for ticker, cik in COMPANIES.items():
        try:
            data = get_submissions_json(cik)
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accs  = recent.get("accessionNumber", [])
            prims = recent.get("primaryDocument", [])
            dates = recent.get("filingDate", [])

            # iterate recent filings and pick 10-K for target YEARS
            for form, acc, prim, fdate in zip(forms, accs, prims, dates):
                if form != "10-K":
                    continue
                year = int(fdate.split("-")[0])
                if year not in YEARS:
                    continue
                try:
                    text = fetch_primary_doc(cik, acc, prim)
                    # filter out tiny wrapper files
                    if len(text) < 5000:
                        # sometimes primaryDocument is a small index; try a fallback guess (often '10k.htm' or similar)
                        # but to stay robust, we keep it if > 1000 chars; else skip
                        if len(text) < 1000:
                            print(f"[skip-small] {ticker} {acc}/{prim} ({len(text)} chars)")
                            continue
                    out_rows.append({
                        "doc_id": acc,
                        "company": ticker,
                        "year": year,
                        "text": text
                    })
                    print(f"{ticker} {year} {acc} ({len(text)} chars) primary={prim}")
                    # be polite to SEC
                    time.sleep(0.5)
                except requests.HTTPError as e:
                    print(f"[warn] fetch doc failed {ticker} {acc} {prim}: {e}")
                except Exception as e:
                    print(f"[warn] parse doc failed {ticker} {acc} {prim}: {e}")
        except requests.HTTPError as e:
            print(f"[warn] submissions fetch failed {ticker}/{cik}: {e}")
        except Exception as e:
            print(f"[warn] submissions parse failed {ticker}/{cik}: {e}")
    return out_rows

def main():
    rows = collect_10k_docs()
    with open(DOCS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id","company","year","text"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} docs to {DOCS_CSV}")

if __name__ == "__main__":
    main()
