# scripts/build_eval_set.py
import os, sys, re, json, pickle
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE = Path(__file__).resolve().parents[1]
PKL = BASE / "artifacts" / "chunks.pkl"
OUT = BASE / "datasets" / "qa_driver.jsonl"

REV = re.compile(r"\b(revenue|sales)\b", re.I)
DRV = re.compile(r"(driven by|led by|primarily by|resulting from|due to|because of)", re.I)

def sentences(text: str):
    for s in re.split(r"[\.!?•;]\s+", text):
        s = s.strip()
        if s:
            yield s

def clean_phrase(x: str, max_words=8):
    x = re.sub(r"\s+", " ", x).strip(" ,.;:-")
    words = x.lower().split()
    return " ".join(words[:max_words])

def extract_phrase(sent: str):
    # priority order
    pats = [
        r"driven by ([^.]+?)(?:[,.;]| and )",
        r"led by ([^.]+?)(?:[,.;]| and )",
        r"primarily by ([^.]+?)(?:[,.;]| and )",
        r"resulting from ([^.]+?)(?:[,.;]| and )",
        r"due to ([^.]+?)(?:[,.;]| and )",
        r"because of ([^.]+?)(?:[,.;]| and )",
    ]
    for p in pats:
        m = re.search(p, sent, re.I)
        if m:
            return clean_phrase(m.group(1))
    return None

def main(limit_per_company=10):
    os.makedirs(OUT.parent, exist_ok=True)
    data = pickle.load(open(PKL, "rb"))
    records = []
    seen = set()

    for text, meta in zip(data["chunks"], data["meta"]):
        comp = meta["company"]
        for s in sentences(text):
            if REV.search(s) and DRV.search(s):
                phrase = extract_phrase(s)
                if not phrase:
                    continue
                key = (comp, phrase)
                if key in seen:
                    continue
                seen.add(key)
                rec = {
                    "company": comp,
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "question": "Which product or service was revenue growth driven by?",
                    "answer": phrase,
                    "sentence": s.strip(),
                }
                records.append(rec)

    # keep a small balanced set per company
    by_co = {}
    for r in records:
        by_co.setdefault(r["company"], []).append(r)
    trimmed = []
    for co, rs in by_co.items():
        trimmed.extend(rs[:limit_per_company])

    with open(OUT, "w") as f:
        for r in trimmed:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(trimmed)} items -> {OUT}")

if __name__ == "__main__":
    main()
