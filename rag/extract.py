import re
from typing import Iterable, Optional, Tuple

# Priority: strongest first
_PATTERNS = [
    (r"driven by ([^.]+?)(?:[,.;]| and )", 3.0),
    (r"led by ([^.]+?)(?:[,.;]| and )", 2.8),
    (r"primarily by ([^.]+?)(?:[,.;]| and )", 2.5),
    (r"resulting from ([^.]+?)(?:[,.;]| and )", 2.0),
    (r"due to ([^.]+?)(?:[,.;]| and )", 1.5),
    (r"because of ([^.]+?)(?:[,.;]| and )", 1.5),
]

# sentence must mention revenue/sales + a change verb to qualify
_SENT_INCLUDE = re.compile(r"\b(revenue|sales)\b", re.I)
_SENT_CHANGE  = re.compile(r"\b(increased|grew|rose|declined|decreased|was up|was down)\b", re.I)

def _sentences(text: str) -> Iterable[str]:
    # also split on bullets and semicolons to catch 10-K formatting
    for s in re.split(r"[\.!?•;]\s+", text):
        s = s.strip()
        if s:
            yield s

def _clean_phrase(x: str, max_words: int = 8) -> str:
    x = re.sub(r"\s+", " ", x).strip(" ,.;:-").lower()
    words = x.split()
    return " ".join(words[:max_words])

def extract_driver(texts: Iterable[str]) -> Optional[str]:
    best: Tuple[float, str] = (0.0, "")
    for t in texts:
        for s in _sentences(t):
            if not (_SENT_INCLUDE.search(s) and _SENT_CHANGE.search(s)):
                continue
            for pat, w in _PATTERNS:
                m = re.search(pat, s, flags=re.I)
                if m:
                    phrase = _clean_phrase(m.group(1))
                    # boost if phrase contains product/service-like words
                    bonus = 0.3 if re.search(r"\b(azure|iphone|windows|microsoft 365|office|search|advertising|cloud|services|hardware|gaming)\b", phrase, re.I) else 0.0
                    score = w + bonus
                    if score > best[0]:
                        best = (score, phrase)
    return best[1] if best[0] > 0 else None
