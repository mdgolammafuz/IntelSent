
# Build a tiny eval set of (question, company, year, category, answer, chunk_id)
from __future__ import annotations

import os, sys, json, argparse, random, re
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE = Path(__file__).resolve().parents[1]
CHUNKS = BASE / "datasets" / "sec_chunks.csv"
OUT = BASE / "datasets" / "qa_driver.jsonl"

# simple patterns to propose QA
DRV = re.compile(r"(driven by|led by|primarily by|resulting from|due to|because of)", re.I)
REV = re.compile(r"\b(revenue|sales)\b", re.I)
MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
OPEX = re.compile(r"\b(operating expenses|opex|research and development|sales and marketing|general and administrative)\b", re.I)

def load_chunks() -> pd.DataFrame:
    if not CHUNKS.exists():
        raise FileNotFoundError(f"{CHUNKS} not found. Run data/chunker.py first.")
    df = pd.read_csv(CHUNKS)
    # expected columns: chunk_id, doc_id, company, year, text, ...
    return df

def pick_examples(df: pd.DataFrame, companies, years, max_per_cat=6):
    out = []

    def add_example(row, category, answer):
        q = {
            "revenue_driver": "Which product or service was revenue growth driven by?",
            "margin_driver": "What primarily improved gross margin?",
            "opex_driver": "What primarily increased operating expenses?",
        }[category]
        out.append({
            "question": q,
            "company": row["company"],
            "year": int(row["year"]),
            "category": category,
            "answer": answer.lower().strip(),
            "chunk_id": int(row["chunk_id"]),
            "doc_id": row["doc_id"],
        })

    df = df[df["company"].isin([c.upper() for c in companies])]
    if years:
        df = df[df["year"].astype(int).isin(years)]

    # Heuristics to seed answers for the eval set
    for _, r in df.iterrows():
        txt = str(r["text"])
        if not DRV.search(txt):
            continue
        tl = txt.lower()
        if REV.search(tl):
            # pick known phrases if present
            for key in ["azure", "office 365", "search", "xbox game pass", "iphone"]:
                if key in tl:
                    add_example(r, "revenue_driver", key)
        if MARGIN.search(tl):
            for key in ["improvement across our cloud services", "improvement in productivity", "improvement in azure"]:
                if key in tl:
                    add_example(r, "margin_driver", key)
        if OPEX.search(tl):
            for key in ["headcount - related expenses"]:
                if key in tl:
                    add_example(r, "opex_driver", key)

    # de-dupe by (company,year,category,answer)
    seen = set()
    uniq = []
    for e in out:
        k = (e["company"], e["year"], e["category"], e["answer"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)

    # cap per category
    by_cat = {}
    final = []
    for e in uniq:
        k = (e["company"], e["year"], e["category"])
        by_cat.setdefault(k, 0)
        if by_cat[k] < max_per_cat:
            final.append(e)
            by_cat[k] += 1

    random.shuffle(final)
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", nargs="+", default=["AAPL", "MSFT"])
    ap.add_argument("--years", nargs="+", type=int, default=[])
    ap.add_argument("--max_per_cat", type=int, default=6)
    args = ap.parse_args()

    df = load_chunks()
    items = pick_examples(df, args.companies, args.years, args.max_per_cat)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(items)} items -> {OUT}")

if __name__ == "__main__":
    main()
