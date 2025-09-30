from __future__ import annotations

import os
import re
import pickle
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CHUNKS_PKL = os.path.join(ARTIFACTS_DIR, "chunks.pkl")
DRIVERS_CFG_PATH = os.path.join(BASE_DIR, "config", "drivers.yml")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _load_corpus() -> List[Dict[str, Any]]:
    """
    Accept multiple shapes for artifacts/chunks.pkl:

      1) {"chunks": List[str], "meta": List[dict]}
      2) List[{"text": str, "meta": dict, ...}]
      3) List[Tuple[str, dict]] / List[List[str, dict]]
      4) List[{"chunk"/"content": str, ...}]  -> meta = rest

    Returns a normalized list of {"text": str, "meta": dict}.
    """
    with open(CHUNKS_PKL, "rb") as f:
        obj = pickle.load(f)

    # Case 1: dict with explicit keys
    if isinstance(obj, dict):
        if "chunks" in obj and "meta" in obj and isinstance(obj["chunks"], list) and isinstance(obj["meta"], list):
            return [{"text": t, "meta": m} for t, m in zip(obj["chunks"], obj["meta"])]
        if "records" in obj and isinstance(obj["records"], list):
            obj = obj["records"]  # fallthrough

    # Case 2: list of dicts / tuples
    if isinstance(obj, list):
        if not obj:
            return []

        # 2a) list of dicts
        if isinstance(obj[0], dict):
            out: List[Dict[str, Any]] = []
            for r in obj:
                if not isinstance(r, dict):
                    continue
                text = (
                    r.get("text")
                    or r.get("chunk")
                    or r.get("content")
                    or next((v for v in r.values() if isinstance(v, str)), "")
                )
                meta = r.get("meta")
                if meta is None:
                    meta = {k: v for k, v in r.items() if k not in {"text", "chunk", "content"}}
                out.append({"text": text or "", "meta": meta or {}})
            return out

        # 2b) list of tuples/lists: [(text, meta), ...]
        if isinstance(obj[0], (tuple, list)) and len(obj[0]) >= 2:
            return [{"text": t, "meta": (m if isinstance(m, dict) else {})} for (t, m, *_) in obj]

    raise RuntimeError("Unsupported format in artifacts/chunks.pkl")


def _iter_company_year(
    company: Optional[str] = None,
    year: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    """
    Yield normalized records filtered by company/year if provided.
    """
    target_company = (company or "").strip().lower() if company else None
    target_year = int(year) if year is not None else None

    for rec in _load_corpus():
        text = (rec.get("text") or "").strip()
        meta = rec.get("meta") or {}

        m_company = (meta.get("company") or meta.get("Company") or meta.get("ticker") or "")
        m_year = meta.get("year") or meta.get("Year")

        if target_company:
            if str(m_company).strip().lower() != target_company:
                continue
        if target_year is not None:
            try:
                if int(m_year) != target_year:
                    continue
            except Exception:
                continue

        yield {"text": text, "meta": meta}


def _default_patterns(cfg: Dict[str, Any]) -> List[str]:
    return cfg.get("driver_regex") or [
        r"(?:driven by|led by|due to)\s+([^.;:\n]+)"
    ]


def _segment_keywords(cfg: Dict[str, Any]) -> List[str]:
    kws: List[str] = []
    for seg_words in (cfg.get("segments") or {}).values():
        for w in seg_words:
            kws.append(w.lower())
    return kws


# --------------------------------------------------------------------------------------
# Public API used by tests
# --------------------------------------------------------------------------------------
def find_revenue_driver(
    company: str,
    year: Optional[int] = None,
    patterns: Optional[List[str]] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Return a dict matching tests' expectations:
      {"answer": <short phrase>, "context": <full text>, "meta": {...}}
    If a regex match isn't found, fall back to a slice of the best context.
    """
    cfg = _load_yaml(DRIVERS_CFG_PATH)
    pat_list = patterns or _default_patterns(cfg)
    regexes = [re.compile(p, re.IGNORECASE) for p in pat_list]
    keywords = set(_segment_keywords(cfg))

    # Collect candidate contexts (bias toward those with known keywords)
    cands: List[Tuple[int, Dict[str, Any]]] = []
    for rec in _iter_company_year(company=company, year=year):
        t = rec["text"]
        score = sum(1 for w in keywords if w in t.lower()) if keywords else 0
        cands.append((score, rec))

    # Prefer higher scores, keep top_k
    cands.sort(key=lambda x: x[0], reverse=True)
    cands = cands[: max(top_k, 8)] if cands else []

    # Try regex extraction
    for _, rec in cands:
        txt = rec["text"]
        for rx in regexes:
            m = rx.search(txt)
            if m:
                phrase = m.group(1).strip()
                phrase = re.sub(r"[\s·•]+$", "", phrase)
                if phrase:
                    return {"answer": phrase, "context": txt, "meta": rec.get("meta", {})}

    # Fallback: return the first candidate's snippet so tests don't get None
    if cands:
        txt = cands[0][1]["text"]
        snippet = (txt[:280] + "…") if len(txt) > 280 else txt
        return {"answer": snippet, "context": txt, "meta": cands[0][1].get("meta", {})}

    # No data at all
    return {"answer": "", "context": "", "meta": {}}
