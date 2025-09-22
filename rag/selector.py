
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from joblib import load

# Driver patterns and anchors
_DRV_PATTS = [
    re.compile(r"driven by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"led by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"primarily by ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"resulting from ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"due to ([^.]+?)(?:[,.;]| and )", re.I),
    re.compile(r"because of ([^.]+?)(?:[,.;]| and )", re.I),
]
_SENT_SPLIT = re.compile(r"[\.!?•;]\s+")
_REV = re.compile(r"\b(revenue|sales)\b", re.I)
_MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
_OPEX = re.compile(r"\b(operating expenses|opex|research and development|r&d|sales and marketing|s&m|general and administrative|g&a)\b", re.I)

SUBJECT_GROUPS = {
    "search": re.compile(r"\b(search|search and news advertising|advertising)\b", re.I),
    "office": re.compile(r"\b(office 365|microsoft 365|office|office commercial)\b", re.I),
    "xbox": re.compile(r"\b(xbox|game pass|xbox game pass)\b", re.I),
    "windows": re.compile(r"\b(windows|windows oem|windows commercial)\b", re.I),
    "cloud": re.compile(r"\b(cloud services|microsoft cloud|server products|intelligent cloud)\b", re.I),
    "iphone": re.compile(r"\b(iphone)\b", re.I),
}
CAND_GROUPS = {
    "search": re.compile(r"\b(search|search and news advertising|advertising)\b", re.I),
    "office": re.compile(r"\b(office 365|microsoft 365|office commercial)\b", re.I),
    "xbox": re.compile(r"\b(xbox|game pass|xbox game pass)\b", re.I),
    "windows": re.compile(r"\b(windows|windows oem|windows commercial)\b", re.I),
    "cloud": re.compile(r"\b(azure|cloud services|microsoft cloud|server products)\b", re.I),
    "iphone": re.compile(r"\b(iphone)\b", re.I),
}

def _sentences(text: str):
    for s in _SENT_SPLIT.split(text or ""):
        s = s.strip()
        if s:
            yield s

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).strip(" ,.;:-").lower()

def _tokens(s: str):
    return [(m.start(), m.end()) for m in re.finditer(r"\w+|\S", s)]

def _char_to_tok(toks, ch: int) -> int:
    for i, (a, b) in enumerate(toks):
        if a <= ch < b:
            return i
    return max(0, len(toks) - 1)

def _nearest_anchor_within(sentence: str, anchor: re.Pattern, span: Tuple[int,int], win: int) -> bool:
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

def _anchor_ok(sentence: str, category: Optional[str], span: Tuple[int,int]) -> bool:
    if category == "revenue_driver":
        return _nearest_anchor_within(sentence, _REV, span, 8)
    if category == "margin_driver":
        return _nearest_anchor_within(sentence, _MARGIN, span, 12)
    if category == "opex_driver":
        return _nearest_anchor_within(sentence, _OPEX, span, 12)
    return True

def _subject_groups(pre_clause: str):
    return [k for k, rx in SUBJECT_GROUPS.items() if rx.search(pre_clause)]

def _candidate_groups(phrase: str):
    return [k for k, rx in CAND_GROUPS.items() if rx.search(phrase)]

@dataclass
class Candidate:
    phrase: str
    sentence: str
    ctx_rank: int
    ce_score: float
    subj_groups: List[str]
    cand_groups: List[str]

def extract_candidates(docs: List[Dict[str, Any]], category: Optional[str]) -> List[Candidate]:
    out: List[Candidate] = []
    for i, d in enumerate(docs):
        text = d.get("text", "")
        ce = float(d.get("ce_score", 0.0) or 0.0)
        for sent in _sentences(text):
            for patt in _DRV_PATTS:
                m = patt.search(sent)
                if not m:
                    continue
                phrase = _clean(m.group(1))
                span = m.span(1)
                if not _anchor_ok(sent, category, span):
                    continue
                subj = sent[: m.start()]
                out.append(
                    Candidate(
                        phrase=phrase,
                        sentence=sent,
                        ctx_rank=i,
                        ce_score=ce,
                        subj_groups=_subject_groups(subj),
                        cand_groups=_candidate_groups(phrase),
                    )
                )
    return out

# ---------- Learned selector ----------
class LearnedSelector:
    def __init__(self, model_path: str | Path):
        p = Path(model_path)
        if not p.exists():
            self.model = None
            self.feature_names = []
        else:
            obj = load(p)
            self.model = obj["model"]
            self.feature_names = obj["feature_names"]

    def _featurize(self, c: Candidate, category: Optional[str]) -> Dict[str, float]:
        feats: Dict[str, float] = {
            "bias": 1.0,
            "ctx_rank": c.ctx_rank,
            "ctx_rank_inv": 1.0 / (1 + c.ctx_rank),
            "ce_score": c.ce_score,
            "len_words": len(c.phrase.split()),
            "subj_any": 1.0 if c.subj_groups else 0.0,
            "g_search": 1.0 if "search" in c.cand_groups else 0.0,
            "g_office": 1.0 if "office" in c.cand_groups else 0.0,
            "g_xbox": 1.0 if "xbox" in c.cand_groups else 0.0,
            "g_windows": 1.0 if "windows" in c.cand_groups else 0.0,
            "g_cloud": 1.0 if "cloud" in c.cand_groups else 0.0,
            "g_iphone": 1.0 if "iphone" in c.cand_groups else 0.0,
            "align": 1.0 if any(g in c.cand_groups for g in c.subj_groups) else 0.0,
            "cat_rev": 1.0 if category == "revenue_driver" else 0.0,
            "cat_margin": 1.0 if category == "margin_driver" else 0.0,
            "cat_opex": 1.0 if category == "opex_driver" else 0.0,
        }
        return feats

    def select(self, candidates: List[Candidate], category: Optional[str], group_hint: Optional[str] = None) -> Optional[str]:
        if not candidates:
            return None

        pool = candidates

        # 1) If we have a group hint, filter to that group first
        if group_hint:
            filt = [c for c in pool if group_hint in c.cand_groups]
            if filt:
                pool = filt

        # 2) For revenue, if subject groups exist, require alignment
        if category == "revenue_driver":
            aligned = [c for c in pool if c.subj_groups and any(g in c.cand_groups for g in c.subj_groups)]
            if aligned:
                pool = aligned

        # 3) No model? heuristic sort
        if self.model is None:
            pool = sorted(
                pool,
                key=lambda c: (
                    any(g in c.cand_groups for g in c.subj_groups),
                    c.ce_score,
                    -c.ctx_rank,
                ),
                reverse=True,
            )
            return pool[0].phrase

        # 4) Model ranking within the filtered pool
        import numpy as np
        X = []
        for c in pool:
            feats = self._featurize(c, category)
            X.append([feats.get(name, 0.0) for name in self.feature_names])
        probs = self.model.predict_proba(np.array(X))[:, 1]
        best = int(probs.argmax())
        return pool[best].phrase

def load_selector(model_path: str | Path) -> LearnedSelector:
    return LearnedSelector(model_path)
