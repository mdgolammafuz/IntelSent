import re
from typing import Iterable, Optional

# Sentence split and phrase extraction
_SENT_SPLIT = re.compile(r"[\.!?•;]\s+")
_DRV_PATTS = [
    re.compile(r"driven by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"led by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"primarily by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"resulting from ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"due to ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"because of ([^.]+?)(?:[,.;]| and )", re.I),
]

# Category anchors
_REV = re.compile(r"\b(revenue|sales)\b", re.I)
_MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
_OPEX = re.compile(
    r"\b(operating expenses|opex|research and development|r&d|sales and marketing|s&m|general and administrative|g&a)\b",
    re.I,
)

# Allowlists / denylists
ALLOW_REVENUE = re.compile(
    r"\b("
    r"azure|office 365|microsoft 365|windows|windows oem|search|news advertising|advertising|"
    r"dynamics 365|xbox|xbox game pass|surface|linkedin|cloud services|server products|"
    r"iphone|mac|ipad|services|wearables|accessories|app store|icloud"
    r")\b",
    re.I,
)
ALLOW_MARGIN = re.compile(
    r"\b(mix|price|pricing|cost|costs|efficien|utili[sz]ation|improvement|azure|cloud services|sales mix)\b",
    re.I,
)
ALLOW_OPEX = re.compile(
    r"\b(headcount|compensation|stock\-based|marketing|sales|r&d|research|engineering|cloud engineering|g&a|general and administrative)\b",
    re.I,
)
DENY_REVENUE = re.compile(r"\b(earthquake|pandemic|supplier|constraint|foreign currency|fx|high cost structure)\b", re.I)


def _sentences(text: str):
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if s:
            yield s


def _clean(phrase: str, max_words: int = 8) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:-")
    return " ".join(phrase.lower().split()[:max_words])


def _anchor_ok(sentence: str, category: Optional[str]) -> bool:
    t = sentence.lower()
    if category is None:
        return True
    if category == "revenue_driver":
        return bool(_REV.search(t))
    if category == "margin_driver":
        return bool(_MARGIN.search(t))
    if category == "opex_driver":
        return bool(_OPEX.search(t))
    return True


def _allowed_for_category(phrase: str, category: Optional[str]) -> bool:
    if category is None:
        return True
    if category == "revenue_driver":
        if DENY_REVENUE.search(phrase):
            return False
        return bool(ALLOW_REVENUE.search(phrase))
    if category == "margin_driver":
        return bool(ALLOW_MARGIN.search(phrase))
    if category == "opex_driver":
        return bool(ALLOW_OPEX.search(phrase))
    return True


def _extract_from_sentence(sentence: str, category: Optional[str]) -> Optional[str]:
    if not _anchor_ok(sentence, category):
        return None
    for p in _DRV_PATTS:
        m = p.search(sentence)
        if not m:
            continue
        cand = _clean(m.group(1))
        if _allowed_for_category(cand, category):
            return cand
    return None


def extract_driver(contexts: Iterable[str], category: Optional[str] = None) -> Optional[str]:
    """
    Extract a short driver phrase from contexts.
    If `category` is provided (revenue_driver / margin_driver / opex_driver),
    we stay STRICT to that category (no cross-category fallback).
    """
    # Pass 1: strict to the given category
    for ctx in contexts:
        for s in _sentences(ctx):
            ans = _extract_from_sentence(s, category)
            if ans:
                return ans

    # Pass 2: only if NO category provided, do generic pattern-only fallback
    if category is None:
        for ctx in contexts:
            for s in _sentences(ctx):
                for p in _DRV_PATTS:
                    m = p.search(s)
                    if m:
                        return _clean(m.group(1))

    return None
