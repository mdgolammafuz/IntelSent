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

# Groups to detect in the SUBJECT (the clause before "driven by")
SUBJECT_GROUPS: Dict[str, re.Pattern] = {
    "search": re.compile(r"\b(search|search and news advertising|advertising)\b", re.I),
    "office": re.compile(r"\b(office 365|microsoft 365|office|office commercial)\b", re.I),
    "xbox": re.compile(r"\b(xbox|game pass|xbox game pass)\b", re.I),
    "windows": re.compile(r"\b(windows|windows oem|windows commercial)\b", re.I),
    "cloud": re.compile(r"\b(cloud services|microsoft cloud|server products|intelligent cloud)\b", re.I),
    "iphone": re.compile(r"\b(iphone)\b", re.I),
}

# What phrases indicate those same groups (in the CANDIDATE)
CAND_GROUPS: Dict[str, re.Pattern] = {
    "search": re.compile(r"\b(search|search and news advertising|advertising)\b", re.I),
    "office": re.compile(r"\b(office 365|microsoft 365|office commercial)\b", re.I),
    "xbox": re.compile(r"\b(xbox|game pass|xbox game pass)\b", re.I),
    "windows": re.compile(r"\b(windows|windows oem|windows commercial)\b", re.I),
    "cloud": re.compile(r"\b(azure|cloud services|microsoft cloud|server products)\b", re.I),
    "iphone": re.compile(r"\b(iphone)\b", re.I),
}

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
    for m in re.finditer(r"\w+|\S", s):
        a, b = m.span()
        toks.append((a, b, m.group(0)))
    return toks

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
    # Tighter window for revenue to avoid margin bleed
    if category == "revenue_driver":
        return _nearest_anchor_within(sentence, _REV, span, win=8)
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

def _score_phrase(phrase: str, category: Optional[str], ctx_rank: int, subj_groups: List[str]) -> float:
    # Lower weight for context position so later good matches can win
    score = 1.0 - min(ctx_rank * 0.05, 0.5)  # 1.0, 0.95, 0.90, ...
    # Group alignment: if subject mentions "search", favor a candidate with "search"
    cand_groups = _candidate_groups(phrase)
    if subj_groups and cand_groups:
        if any(g in cand_groups for g in subj_groups):
            score += 0.8
        # If subject is not cloud but candidate is cloud-only, slight penalty
        if "cloud" in cand_groups and all(g != "cloud" for g in subj_groups):
            score -= 0.4

    if category == "revenue_driver":
        # Prefer product-first phrases; penalize margin-ish words strongly
        if re.match(r"^(azure|office|microsoft|windows|search|xbox|game pass|iphone|surface|linkedin|cloud|server)", phrase):
            score += 0.3
        if re.search(r"\b(margin|improvement|cost|pricing|mix)\b", phrase):
            score -= 0.7
    return score

# -------- Main API --------
def extract_driver(contexts: Iterable[str], category: Optional[str] = None) -> Optional[str]:
    """
    Extract a short 'driver' phrase from contexts.
    - STRICT to the given category (no cross-category fallback).
    - Requires proximity: the category anchor (e.g., 'revenue') must be within N tokens of the driver phrase.
    - Scores candidates across top-k contexts with subject-aware boosting.
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

                if not _anchor_proximity_ok(sent, category, span):
                    continue
                if not _allowed_for_category(phrase, category):
                    continue

                # Subject groups: words before "driven by"
                subj = sent[: m.start()]
                subj_groups = _subject_groups(subj)

                score = _score_phrase(phrase, category, ctx_rank=ctx_i, subj_groups=subj_groups)
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
