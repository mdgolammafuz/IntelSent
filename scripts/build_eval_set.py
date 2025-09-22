import os
import sys
import re
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE = Path(__file__).resolve().parents[1]
PKL = BASE / "artifacts" / "chunks.pkl"
OUT = BASE / "datasets" / "qa_driver.jsonl"

# Patterns
DRV_PATTS = [
    re.compile(r"driven by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"led by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"primarily by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"resulting from ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"due to ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"because of ([^.]+?)(?:[,.;]| and )", re.I),
]
REV = re.compile(r"\b(revenue|sales)\b", re.I)
MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
OPEX = re.compile(
    r"\b(operating expenses|opex|research and development|r&d|sales and marketing|s&m|general and administrative|g&a)\b",
    re.I,
)

# Allow/deny filters for cleaner labels
ALLOW_REVENUE = re.compile(
    r"\b(azure|office 365|microsoft 365|windows|search|advertising|dynamics 365|xbox|surface|"
    r"linkedin|cloud services|server products|iphone|mac|ipad|services|wearables|accessories|app store|icloud)\b",
    re.I,
)
DENY_REVENUE = re.compile(r"\b(earthquake|pandemic|supplier|constraint|foreign currency|fx|high cost structure)\b", re.I)

SENT_SPLIT = re.compile(r"[\.!?•;]\s+")


def sentences(text: str):
    for s in SENT_SPLIT.split(text):
        s = s.strip()
        if s:
            yield s


def tokens(s: str) -> List[str]:
    return re.findall(r"\w+|\S", s.lower())


def within_window(s: str, anchor: re.Pattern, driver_span: Tuple[int, int], win: int = 12) -> bool:
    """
    True if an anchor word (e.g., 'revenue') appears within `win` tokens of the driver phrase span.
    """
    toks = tokens(s)
    # map char offsets to token indices roughly by cumulative length
    idxs = []
    pos = 0
    for i, t in enumerate(toks):
        idxs.append((i, pos, pos + len(t)))
        pos += len(t)

    # driver span char -> token index range
    d0, d1 = driver_span
    d_start_tok = next((i for i, a, b in idxs if a <= d0 < b), 0)
    d_end_tok = next((i for i, a, b in idxs if a < d1 <= b), len(toks) - 1)

    # find first anchor match
    for m in anchor.finditer(s):
        a0, a1 = m.span()
        a_tok = next((i for i, a, b in idxs if a <= a0 < b), 0)
        if abs(a_tok - d_start_tok) <= win or abs(a_tok - d_end_tok) <= win:
            return True
    return False


def extract_phrase(sent: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    for p in DRV_PATTS:
        m = p.search(sent)
        if m:
            phrase = re.sub(r"\s+", " ", m.group(1)).strip(" ,.;:-").lower()
            return phrase, m.span(1)
    return None


def categorize(sent: str, phrase: str, span: Tuple[int, int]) -> Optional[str]:
    # Use proximity windows so the category anchor is near the driver phrase
    if within_window(sent, MARGIN, span, win=12):
        return "margin_driver"
    if within_window(sent, OPEX, span, win=12):
        return "opex_driver"
    if within_window(sent, REV, span, win=12):
        # Revenue: also require allowed terms and reject known non-revenue drivers
        if DENY_REVENUE.search(phrase):
            return None
        if not ALLOW_REVENUE.search(phrase):
            return None
        return "revenue_driver"
    return None


def main(limit_per_company_per_cat: int = 10):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.load(open(PKL, "rb"))

    seen = set()
    items = []

    for text, meta in zip(data["chunks"], data["meta"]):
        comp = meta["company"]
        for s in sentences(text):
            ex = extract_phrase(s)
            if not ex:
                continue
            phrase, span = ex
            cat = categorize(s, phrase, span)
            if not cat:
                continue
            key = (comp, cat, phrase)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "company": comp,
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "category": cat,
                    "question": {
                        "revenue_driver": "Which product or service was revenue growth driven by?",
                        "margin_driver": "What factor drove gross margin change?",
                        "opex_driver": "What primarily drove operating expenses?",
                    }[cat],
                    "answer": phrase,
                    "sentence": s.strip(),
                }
            )

    # Balance per company per category
    buckets = {}
    for it in items:
        k = (it["company"], it["category"])
        buckets.setdefault(k, []).append(it)

    trimmed = []
    for k, lst in buckets.items():
        trimmed.extend(lst[:limit_per_company_per_cat])

    with open(OUT, "w") as f:
        for r in trimmed:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(trimmed)} items -> {OUT}")


if __name__ == "__main__":
    main()
