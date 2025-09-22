import re
from typing import Iterable, Optional, Tuple, List

# -------- Sentence and pattern basics --------
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

# Extra keyword boosts for ranking phrases (for revenue)
KEYWORD_BOOSTS = [
    re.compile(r"\bazure\b", re.I),
    re.compile(r"\boffice 365\b", re.I),
    re.compile(r"\bmicrosoft 365\b", re.I),
    re.compile(r"\bsearch\b", re.I),
    re.compile(r"\bxbox game pass\b", re.I),
    re.compile(r"\bxbox\b", re.I),
    re.compile(r"\bwindows\b", re.I),
    re.compile(r"\biphone\b", re.I),
    re.compile(r"\bcloud services\b", re.I),
]

# -------- Small helpers --------
def _sentences(text: str):
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if s:
            yield s

def _clean(phrase: str, max_words: int = 8) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:-")
    return " ".join(phrase.lower().split()[:max_words])

def _tokens(s: str) -> List[Tuple[int, int, str]]:
    toks = []
    pos = 0
    for m in re.finditer(r"\w+|\S", s):
        t = m.group(0)
        a, b = m.span()
        toks.append((a, b, t))
        pos = b
    return toks

def _nearest_anchor_within(sentence: str, anchor: re.Pattern, span: Tuple[int, int], win: int) -> bool:
    """
    True if an anchor (e.g., 'revenue') appears within `win` tokens of the driver phrase span.
    """
    toks = _tokens(sentence)
    # map char->token indices
    def char_to_tok(ch: int) -> int:
        for i, (a, b, _) in enumerate(toks):
            if a <= ch < b:
                return i
        return max(0, len(toks) - 1)

    d0, d1 = span
    d_start = char_to_tok(d0)
    d_end = char_to_tok(d1 - 1)

    for m in anchor.finditer(sentence):
        a0, _ = m.span()
        a_tok = char_to_tok(a0)
        if abs(a_tok - d_start) <= win or abs(a_tok - d_end) <= win:
            return True
    return False

def _anchor_proximity_ok(sentence: str, category: Optional[str], span: Tuple[int, int], win: int = 12) -> bool:
    if category is None:
        return True
    if category == "revenue_driver":
        return _nearest_anchor_within(sentence, _REV, span, win)
    if category == "margin_driver":
        return _nearest_anchor_within(sentence, _MARGIN, span, win)
    if category == "opex_driver":
        return _nearest_anchor_within(sentence, _OPEX, span, win)
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

def _score_phrase(phrase: str, category: Optional[str], ctx_rank: int) -> float:
    # Higher score for earlier contexts; boost known product terms
    score = 1.0 / (1 + ctx_rank)  # ctx_rank: 0 is best
    if category == "revenue_driver":
        for rx in KEYWORD_BOOSTS:
            if rx.search(phrase):
                score += 0.5
                break
    return score

# -------- Main API --------
def extract_driver(contexts: Iterable[str], category: Optional[str] = None) -> Optional[str]:
    """
    Extract a short 'driver' phrase from contexts.
    - STRICT to the given category (no cross-category fallback).
    - Requires proximity: the category anchor (e.g., 'revenue') must be within N tokens of the 'driven by' phrase.
    - Picks the best candidate across all top-k contexts using a simple scoring rule.
    """
    candidates: List[Tuple[float, str]] = []  # (score, phrase)

    for ctx_i, ctx in enumerate(contexts):
        for sent in _sentences(ctx):
            for patt in _DRV_PATTS:
                m = patt.search(sent)
                if not m:
                    continue
                phrase = _clean(m.group(1))
                span = m.span(1)

                # category-strict with proximity
                if not _anchor_proximity_ok(sent, category, span, win=12):
                    continue
                if not _allowed_for_category(phrase, category):
                    continue

                score = _score_phrase(phrase, category, ctx_rank=ctx_i)
                candidates.append((score, phrase))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # No fallback when category is provided (stay strict)
    if category is None:
        # Generic fallback: first driver phrase anywhere
        for ctx in contexts:
            for sent in _sentences(ctx):
                for patt in _DRV_PATTS:
                    m = patt.search(sent)
                    if m:
                        return _clean(m.group(1))
    return None
