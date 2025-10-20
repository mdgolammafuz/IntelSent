

"""
Eval runner with two modes:

1) OFFLINE (default recommended): pass --no-llm
   - Does NOT import or call ragas
   - Computes local, deterministic metrics from your /query outputs
   - Safe with zero API keys

2) RAGAS (optional): omit --no-llm
   - Tries to run ragas metrics (you must have a working ragas + LLM setup)
   - Left here for future use, but OFFLINE mode is the reliable path today.
"""

import os
import re
import json
import time
import argparse
import random
import pathlib
from typing import List, Dict, Any, Optional, Tuple

import requests
from datasets import Dataset as HFDataset  # used only in ragas mode

API_DEFAULT = "http://127.0.0.1:8000"

# -------------------------
# Text normalization helpers
# -------------------------
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_STOPWORDS = {
    # very small stopword list for stable token F1
    "the","a","an","and","or","but","for","nor","on","in","at","to","from","by",
    "of","with","as","is","are","was","were","be","been","being","that","this",
    "it","its","their","his","her","our","your","my"
}

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [t for t in normalize_text(s).split(" ") if t and t not in _STOPWORDS and len(t) > 1]

# -------------------------
# Offline metrics
# -------------------------
def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def contains_match(pred: str, gold: str) -> int:
    p = normalize_text(pred)
    g = normalize_text(gold)
    return int(p in g or g in p)

def token_f1(pred: str, gold: str) -> float:
    p = set(tokenize(pred))
    g = set(tokenize(gold))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def retrieval_hit(chunks: List[str], gold: str) -> int:
    g = normalize_text(gold)
    if not g:
        return 0
    for c in chunks:
        if g and g in normalize_text(c):
            return 1
    return 0

def context_hit_rate(chunks: List[str], gold: str) -> float:
    g = normalize_text(gold)
    if not chunks or not g:
        return 0.0
    hits = 0
    for c in chunks:
        if g in normalize_text(c):
            hits += 1
    return hits / max(1, len(chunks))

# -------------------------
# I/O
# -------------------------
def load_qa(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def query_api(api: str, q: str, company: Optional[str], year: Optional[int],
              top_k: int = 5, timeout: int = 60) -> Tuple[str, List[str]]:
    payload: Dict[str, Any] = {"text": q, "top_k": top_k}
    if company:
        payload["company"] = company
    if year:
        payload["year"] = year
    r = requests.post(f"{api}/query", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    pred = data.get("answer", "") or ""
    ctxs = data.get("contexts") or [c.get("text", "") for c in data.get("chunks", [])] or []
    return pred, ctxs

# -------------------------
# OFFLINE evaluation
# -------------------------
def run_offline_eval(qa_items: List[Dict[str, Any]], api: str, timeout: int, sleep_s: float) -> Dict[str, Any]:
    per_item = []
    failed = 0
    for ex in qa_items:
        q = ex.get("question") or ex.get("q") or ""
        gold = ex.get("answer") or ex.get("ground_truth") or ""
        company = ex.get("company")
        year = ex.get("year")
        try:
            pred, ctx = query_api(api, q, company, year, top_k=5, timeout=timeout)
        except Exception as e:
            failed += 1
            pred, ctx = "", []
        item = {
            "question": q,
            "ground_truth": gold,
            "prediction": pred,
            "contexts": ctx,
            "exact_match": exact_match(pred, gold),
            "contains_match": contains_match(pred, gold),
            "token_f1": token_f1(pred, gold),
            "retrieval_hit": retrieval_hit(ctx, gold),
            "context_hit_rate": context_hit_rate(ctx, gold),
        }
        per_item.append(item)
        if sleep_s:
            time.sleep(sleep_s)

    # aggregate
    n = len(per_item)
    def avg(key: str) -> float:
        vals = [float(x[key]) for x in per_item]
        return sum(vals) / n if n else 0.0

    summary = {
        "n_items": n,
        "failed_requests": failed,
        "exact_match": avg("exact_match"),
        "contains_match": avg("contains_match"),
        "token_f1": avg("token_f1"),
        "retrieval_hit": avg("retrieval_hit"),
        "context_hit_rate": avg("context_hit_rate"),
    }
    return {"summary": summary, "items": per_item}

# -------------------------
# Optional: ragas mode
# -------------------------
def run_ragas_eval(qa_items: List[Dict[str, Any]], api: str, timeout: int, sleep_s: float) -> Dict[str, Any]:
    """
    Only used if user omits --no-llm. Requires a working ragas + LLM setup.
    """
    from ragas import evaluate
    # pick a conservative set of metrics; user can extend as they wish
    from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness

    rows = []
    failed = 0
    for ex in qa_items:
        q = ex.get("question") or ex.get("q") or ""
        gold = ex.get("answer") or ex.get("ground_truth") or ""
        company = ex.get("company")
        year = ex.get("year")
        try:
            pred, ctx = query_api(api, q, company, year, top_k=5, timeout=timeout)
        except Exception:
            failed += 1
            pred, ctx = "", []
        rows.append({
            "question": q,
            "answer": pred,
            "contexts": ctx,
            "ground_truth": gold,
            "reference": gold,  # some ragas versions expect 'reference'
        })
        if sleep_s:
            time.sleep(sleep_s)

    ds = HFDataset.from_list(rows)
    metrics = [context_precision, context_recall, answer_relevancy, faithfulness]
    result = evaluate(ds, metrics=metrics)

    # normalize result.scores across ragas versions
    scores: Dict[str, float] = {}
    rs = getattr(result, "scores", None)
    if isinstance(rs, dict):
        for k, v in rs.items():
            val = getattr(v, "score", v)
            try:
                scores[k] = float(val)
            except Exception:
                pass
    elif isinstance(rs, list):
        for s in rs:
            name = getattr(s, "name", None)
            if not name and hasattr(s, "metric") and getattr(s, "metric") is not None:
                name = getattr(s.metric, "name", None)
            if not name:
                name = getattr(s, "key", None) or f"metric_{len(scores)}"
            val = getattr(s, "score", s)
            try:
                scores[name] = float(val)
            except Exception:
                pass

    return {
        "summary": {
            "n_items": len(rows),
            "failed_requests": failed,
            "ragas_scores": scores,
        }
    }

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="path to jsonl QA file")
    ap.add_argument("--out", required=True, help="output JSON report path")
    ap.add_argument("--api", default=API_DEFAULT, help="base URL of the running API server")
    ap.add_argument("--timeout", type=int, default=90, help="HTTP timeout per /query call (s)")
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep between /query calls (s)")
    ap.add_argument("--limit", type=int, default=0, help="limit number of QA items (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle QA before limiting")
    ap.add_argument("--no-llm", dest="no_llm", action="store_true",
                    help="OFFLINE mode: compute local metrics and never touch ragas/LLMs")
    args = ap.parse_args()

    # Ensure output directory exists
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qa_items = load_qa(args.qa)
    if args.shuffle:
        random.shuffle(qa_items)
    if args.limit and args.limit > 0:
        qa_items = qa_items[:args.limit]

    if args.no_llm:
        report = run_offline_eval(qa_items, api=args.api, timeout=args.timeout, sleep_s=args.sleep)
    else:
        report = run_ragas_eval(qa_items, api=args.api, timeout=args.timeout, sleep_s=args.sleep)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Tiny console summary
    if "summary" in report:
        print(json.dumps(report["summary"], indent=2))
    else:
        print("Done.")

if __name__ == "__main__":
    main()
