import os, pickle, re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

BASE = os.path.dirname(os.path.dirname(__file__))
CHUNKS_PKL = os.path.join(BASE, "artifacts", "chunks.pkl")

_SENT_INCLUDE = re.compile(r"\b(revenue|sales)\b", re.I)
_SENT_CHANGE  = re.compile(r"\b(increased|grew|rose|declined|decreased|was up|was down)\b", re.I)

# Strongest cues first; weight guides tie-breaks
_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"driven by ([^.]+?)(?:[,.;]| and )", re.I), 3.0),
    (re.compile(r"led by ([^.]+?)(?:[,.;]| and )",     re.I), 2.8),
    (re.compile(r"primarily by ([^.]+?)(?:[,.;]| and )", re.I), 2.5),
    (re.compile(r"resulting from ([^.]+?)(?:[,.;]| and )", re.I), 2.0),
    (re.compile(r"due to ([^.]+?)(?:[,.;]| and )",     re.I), 1.6),
    (re.compile(r"because of ([^.]+?)(?:[,.;]| and )", re.I), 1.6),
]

_KEYWORDS = re.compile(
    r"\b(azure|iphone|services|cloud|windows|microsoft 365|office|search|advertising|gaming|xbox|hardware)\b",
    re.I,
)

def _sentences(text: str):
    for s in re.split(r"[\.!?•;]\s+", text):
        s = s.strip()
        if s:
            yield s

def _clean(phrase: str, max_words: int = 8) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:-").lower()
    return " ".join(phrase.split()[:max_words])

@lru_cache(maxsize=1)
def _load_corpus():
    data = pickle.load(open(CHUNKS_PKL, "rb"))
    # list of dicts: {"text": str, "meta": {...}}
    return [{"text": t, "meta": m} for t, m in zip(data["chunks"], data["meta"])]

def find_revenue_driver(company: Optional[str] = None) -> Optional[Dict]:
    best = (0.0, "", None)  # (score, phrase, record)
    for rec in _load_corpus():
        if company and rec["meta"]["company"].lower() != company.lower():
            continue
        for s in _sentences(rec["text"]):
            if not (_SENT_INCLUDE.search(s) and _SENT_CHANGE.search(s)):
                continue
            for pat, w in _PATTERNS:
                m = pat.search(s)
                if not m:
                    continue
                phrase = _clean(m.group(1))
                bonus = 0.3 if _KEYWORDS.search(phrase) else 0.0
                score = w + bonus
                if score > best[0]:
                    best = (score, phrase, rec)
    if best[2] is None:
        return None
    rec = best[2]
    out = {
        "answer": best[1],
        "chunk": {
            "chunk_id": rec["meta"]["chunk_id"],
            "doc_id": rec["meta"]["doc_id"],
            "company": rec["meta"]["company"],
            "year": rec["meta"]["year"],
            "text": rec["text"],
        },
    }
    return out
