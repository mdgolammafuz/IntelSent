
"""
Offline evaluator for IntelSent API (no LLMs, no ragas).

Reads a QA JSONL, calls /query, computes simple deterministic metrics:
- exact_match, contains_match, token_f1
- retrieval_hit, context_hit_rate

QA JSONL rows can include: question, ground_truth (or answer), company, year
"""

import argparse, json, pathlib, random, re, time
from typing import Any, Dict, List, Optional, Tuple
import requests

API_DEFAULT = "http://127.0.0.1:8000"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_STOPWORDS = {
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

def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))

def contains_match(pred: str, gold: str) -> int:
    p = normalize_text(pred); g = normalize_text(gold)
    return int(p in g or g in p)

def token_f1(pred: str, gold: str) -> float:
    p = set(tokenize(pred)); g = set(tokenize(gold))
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    tp = len(p & g)
    prec = tp / len(p); rec = tp / len(g)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))

def retrieval_hit(chunks: List[str], gold: str) -> int:
    g = normalize_text(gold)
    return 0 if not g else int(any(g in normalize_text(c) for c in chunks))

def context_hit_rate(chunks: List[str], gold: str) -> float:
    g = normalize_text(gold)
    if not chunks or not g: return 0.0
    hits = sum(1 for c in chunks if g in normalize_text(c))
    return hits / max(1, len(chunks))

def load_qa(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def query_api(api: str, api_key: Optional[str], q: str, company: Optional[str],
              year: Optional[int], top_k: int, use_openai: bool, timeout: int
             ) -> Tuple[str, List[str]]:
    payload: Dict[str, Any] = {"text": q, "top_k": top_k, "no_openai": (not use_openai)}
    if company: payload["company"] = company
    if year: payload["year"] = year
    headers = {"Content-Type": "application/json"}
    if api_key: headers["X-API-Key"] = api_key
    r = requests.post(f"{api}/query", json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    pred = data.get("answer") or ""
    ctxs = data.get("contexts") or [c.get("text","") for c in data.get("chunks",[])] or []
    return pred, ctxs

def run_offline_eval(qa_items: List[Dict[str, Any]], api: str, api_key: Optional[str],
                     timeout: int, sleep_s: float, top_k: int, use_openai: bool) -> Dict[str, Any]:
    per_item = []; failed = 0
    for ex in qa_items:
        q = ex.get("question") or ex.get("q") or ""
        gold = ex.get("ground_truth") or ex.get("answer") or ""
        company = ex.get("company"); year = ex.get("year")
        try:
            pred, ctx = query_api(api, api_key, q, company, year, top_k, use_openai, timeout)
        except Exception:
            failed += 1; pred, ctx = "", []
        item = {
            "question": q, "ground_truth": gold, "prediction": pred, "contexts": ctx,
            "exact_match": exact_match(pred, gold),
            "contains_match": contains_match(pred, gold),
            "token_f1": token_f1(pred, gold),
            "retrieval_hit": retrieval_hit(ctx, gold),
            "context_hit_rate": context_hit_rate(ctx, gold),
        }
        per_item.append(item)
        if sleep_s: time.sleep(sleep_s)

    n = len(per_item)
    def avg(k: str) -> float: return (sum(float(x[k]) for x in per_item)/n) if n else 0.0
    summary = {
        "n_items": n, "failed_requests": failed,
        "exact_match": avg("exact_match"),
        "contains_match": avg("contains_match"),
        "token_f1": avg("token_f1"),
        "retrieval_hit": avg("retrieval_hit"),
        "context_hit_rate": avg("context_hit_rate"),
    }
    return {"summary": summary, "items": per_item}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="path to jsonl QA file")
    ap.add_argument("--out", required=True, help="output JSON report path")
    ap.add_argument("--api", default=API_DEFAULT, help="API base URL")
    ap.add_argument("--api-key", default=None, help="X-API-Key header (if API requires it)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--use-openai", action="store_true", help="call LLM path if API allows")
    args = ap.parse_args()

    qa = load_qa(args.qa)
    if args.shuffle: random.shuffle(qa)
    if args.limit and args.limit > 0: qa = qa[:args.limit]

    report = run_offline_eval(
        qa_items=qa, api=args.api, api_key=args.api_key, timeout=args.timeout,
        sleep_s=args.sleep, top_k=args.top_k, use_openai=args.use_openai
    )

    out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(report, f, indent=2)

    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__":
    main()
