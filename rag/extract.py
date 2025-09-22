import re
from typing import Iterable, Optional, Tuple, List, Dict

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

# Category anchors (must be near the driver phrase)
_REV = re.compile(r"\b(revenue|sales)\b", re.I)
_MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
_OPEX = re.compile(
    r"\b(operating expenses|opex|research and development|r&d|sales and marketing|s&m|general and administrative|g&a)\b",
    re.I,
)

# Revenue allow/deny and synonyms
ALLOW_REVENUE = re.compile(
    r"\b("
    r"azure|office 365|microsoft 365|office commercial|windows|windows oem|windows commercial|"
    r"search and news advertising|search|advertising|dynamics 365|xbox|xbox game pass|game pass|"
    r"surface|linkedin|cloud services|microsoft cloud|server products|"
    r"iphone|mac|ipad|services|wearables|accessories|app store|icloud"
    r")\b",
    re.I,
)
# For revenue, block non-product drivers and margin-ish words
DENY_REVENUE = re.compile(
    r"\b(earthquake|pandemic|supplier|constraint|foreign currency|fx|high cost structure|improvement|margin|cost|mix)\b",
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

# SUBJECT groups detected before "driven by"
SUBJECT_GROUPS: Dict[str, re.Pattern] = {
    "search": re.compile(r"\b(search|search and news advertising|advertising)\b", re.I),
    "office": re.compile(r"\b(office 365|microsoft 365|office|office commercial)\b", re.I),
    "xbox": re.compile(r"\b(xbox|game pass|xbox game pass)\b", re.I),
    "windows": re.compile(r"\b(windows|windows oem|windows commercial)\b", re.I),
    "cloud": re.compile(r"\b(cloud services|microsoft cloud|server products|intelligent cloud)\b", re.I),
    "iphone": re.compile(r"\b(iphone)\b", re.I),
}

# Candidate groups matched inside extracted phrase
CAND_GROUPS: Dict[str, re.Pattern] = {
    "search": re.compile(r"\b(search|search and news advertising|advertising)\b", re.I),
    "office": re.compile(r"\b(office 365|microsoft 365|office commercial)\b", re.I),
    "xbox": re.compile(r"\b(xbox|game pass|xbox game pass)\b", re.I),
    "windows": re.compile(r"\b(windows|windows oem|windows commercial)\b", re.I),
    "cloud": re.compile(r"\b(azure|cloud services|microsoft cloud|server products)\b", re.I),
    "iphone": re.compile(r"\b(iphone)\b", re.I),
}

def _sentences(text: str):
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if s:
            yield s

def _clean(phrase: str, max_words: int = 8) -> str:
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.;:-")
    return " ".join(phrase.lower().split()[:max_words])

def _tokens(s: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\w+|\S", s)]

def _char_to_tok(toks: List[Tuple[int,int,str]], ch: int) -> int:
    for i, (a, b, _) in enumerate(toks):
        if a <= ch < b:
            return i
    return max(0, len(toks) - 1)

def _nearest_anchor_within(sentence: str, anchor: re.Pattern, span: Tuple[int, int], win: int) -> bool:
    toks = _tokens(sentence)
    d0, d1 = span
    d_start = _char_to_tok(toks, d0)
    d_end = _char_to_tok(toks, d1 - 1)
    for m in anchor.finditer(sentence):
        a0, _ = m.span()
        a_tok = _char_to_tok(toks, a0)
        if abs(a_tok - d_start) <= win or abs(a_tok - d_end) <= win:
            return True
    return False

def _anchor_proximity_ok(sentence: str, category: Optional[str], span: Tuple[int, int]) -> bool:
    if category is None:
        return True
    if category == "revenue_driver":
        return _nearest_anchor_within(sentence, _REV, span, win=8)   # tighter for revenue
    if category == "margin_driver":
        return _nearest_anchor_within(sentence, _MARGIN, span, win=12)
    if category == "opex_driver":
        return _nearest_anchor_within(sentence, _OPEX, span, win=12)
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

def _subject_groups(pre_clause: str) -> List[str]:
    groups = []
    for name, rx in SUBJECT_GROUPS.items():
        if rx.search(pre_clause):
            groups.append(name)
    return groups

def _candidate_groups(phrase: str) -> List[str]:
    groups = []
    for name, rx in CAND_GROUPS.items():
        if rx.search(phrase):
            groups.append(name)
    return groups

def _score_phrase(phrase: str, category: Optional[str], ctx_rank: int) -> float:
    # Lower weight for context position so later good matches can win
    return 1.0 - min(ctx_rank * 0.04, 0.4)  # 1.00, 0.96, 0.92, ...

def extract_driver(contexts: Iterable[str], category: Optional[str] = None) -> Optional[str]:
    """
    Extract a short 'driver' phrase from contexts.
    - STRICT to the given category.
    - Requires proximity: category anchor (e.g., 'revenue') close to the driver phrase.
    - Requires SUBJECT↔CANDIDATE group alignment for revenue (e.g., 'search' subject → 'search' candidate).
    - Ranks candidates with light context bias.
    """
    candidates: List[Tuple[float, str]] = []

    for ctx_i, ctx in enumerate(contexts):
        for sent in _sentences(ctx):
            for patt in _DRV_PATTS:
                m = patt.search(sent)
                if not m:
                    continue
                phrase = _clean(m.group(1))
                span = m.span(1)

                if not _anchor_proximity_ok(sent, category, span):
                    continue
                if not _allowed_for_category(phrase, category):
                    continue

                # SUBJECT before "driven by"
                subj_clause = sent[: m.start()]
                subj_groups = _subject_groups(subj_clause)
                cand_groups = _candidate_groups(phrase)

                # Hard alignment for revenue: if subject mentions a group, candidate must match it
                if category == "revenue_driver" and subj_groups:
                    if not any(g in cand_groups for g in subj_groups):
                        continue

                score = _score_phrase(phrase, category, ctx_rank=ctx_i)
                candidates.append((score, phrase))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # No fallback when category is provided (stay strict)
    if category is None:
        for ctx in contexts:
            for sent in _sentences(ctx):
                for patt in _DRV_PATTS:
                    m = patt.search(sent)
                    if m:
                        return _clean(m.group(1))
    return None
