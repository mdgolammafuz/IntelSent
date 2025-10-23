"""
Offline evaluator for IntelSent API (no LLMs, no ragas).

Reads a QA JSONL, calls /query, computes deterministic metrics:
- exact_match, contains_match, token_f1
- retrieval_hit, context_hit_rate
- latency_ms_overall: mean/p50/p95/max (all attempts)
- latency_ms_success_only: mean/p50/p95/max (2xx only)
- request_success_rate
- retrieval metrics (mrr, hit@1/3/5, first_hit_rank)
- length stats (answer_len_tokens, avg_context_len_tokens)

QA JSONL rows can include: question, ground_truth (or answer), company, year
"""

import argparse, json, pathlib, random, re, time
from typing import Any, Dict, List, Optional, Tuple
import requests

API_DEFAULT = "http://127.0.0.1:8000"

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_STOPWORDS = {
    "the","a","an","and","or","but","for","nor","beyond","on","in","at","to","from","by",
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
    if not p or not g:
        return 0
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

def first_hit_rank(chunks: List[str], gold: str) -> Optional[int]:
    """1-based rank of first context containing gold; None if no hit."""
    g = normalize_text(gold)
    if not g: return None
    for i, c in enumerate(chunks, start=1):
        if g in normalize_text(c):
            return i
    return None

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _percentile(values: List[float], q: float) -> float:
    """Nearest-rank with linear interp; q in [0,100]."""
    if not values: return 0.0
    xs = sorted(values)
    if len(xs) == 1: return xs[0]
    pos = (q/100.0) * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac

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
             ) -> Tuple[str, List[str], int]:
    payload: Dict[str, Any] = {"text": q, "top_k": top_k, "no_openai": (not use_openai)}
    if company: payload["company"] = company
    if year: payload["year"] = year
    headers = {"Content-Type": "application/json"}
    if api_key: headers["X-API-Key"] = api_key
    try:
        r = requests.post(f"{api}/query", json=payload, headers=headers, timeout=timeout)
        status = r.status_code
        r.raise_for_status()
        data = r.json()
        pred = data.get("answer") or ""
        ctxs = data.get("contexts") or [c.get("text","") for c in data.get("chunks",[])] or []
        return pred, ctxs, status
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
        return "", [], int(code or 0)
    except Exception:
        return "", [], 0

def run_offline_eval(qa_items: List[Dict[str, Any]], api: str, api_key: Optional[str],
                     timeout: int, sleep_s: float, top_k: int, use_openai: bool) -> Dict[str, Any]:
    per_item: List[Dict[str, Any]] = []
    success_items: List[Dict[str, Any]] = []
    failed = 0

    lat_all: List[float] = []
    lat_ok: List[float] = []

    rr_list_all: List[float] = []
    rr_list_ok: List[float] = []
    hit1_all = hit3_all = hit5_all = 0
    hit1_ok = hit3_ok = hit5_ok = 0

    ans_lens_all: List[float] = []
    ctx_lens_all: List[float] = []
    ans_lens_ok: List[float] = []
    ctx_lens_ok: List[float] = []

    for ex in qa_items:
        q = ex.get("question") or ex.get("q") or ""
        gold = ex.get("ground_truth") or ex.get("answer") or ""
        company = ex.get("company"); year = ex.get("year")

        t0 = time.perf_counter()
        pred, ctx, code = query_api(api, api_key, q, company, year, top_k, use_openai, timeout)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        status_ok = bool(code) and 200 <= code < 300
        status = "ok" if status_ok else "fail"
        if not status_ok: failed += 1

        lat_all.append(dt_ms)
        if status_ok: lat_ok.append(dt_ms)

        rank = first_hit_rank(ctx, gold)
        rr = 1.0 / rank if rank and rank > 0 else 0.0
        rr_list_all.append(rr)
        if status_ok: rr_list_ok.append(rr)
        if rank is not None:
            if rank <= 1: 
                hit1_all += 1
                if status_ok: hit1_ok += 1
            if rank <= 3:
                hit3_all += 1
                if status_ok: hit3_ok += 1
            if rank <= 5:
                hit5_all += 1
                if status_ok: hit5_ok += 1

        ans_len = len(tokenize(pred))
        ctx_len = mean([len(tokenize(c)) for c in ctx]) if ctx else 0.0
        ans_lens_all.append(ans_len)
        ctx_lens_all.append(ctx_len)
        if status_ok:
            ans_lens_ok.append(ans_len)
            ctx_lens_ok.append(ctx_len)

        item = {
            "http_status": code,
            "status": status,
            "latency_ms": round(dt_ms, 3),
            "question": q,
            "ground_truth": gold,
            "prediction": pred,
            "contexts": ctx,
            "first_hit_rank": rank,
            "reciprocal_rank": rr,
            "answer_len_tokens": ans_len,
            "avg_context_len_tokens": ctx_len,
            "exact_match": exact_match(pred, gold),
            "contains_match": contains_match(pred, gold),
            "token_f1": token_f1(pred, gold),
            "retrieval_hit": retrieval_hit(ctx, gold),
            "context_hit_rate": context_hit_rate(ctx, gold),
        }
        per_item.append(item)
        if status_ok:
            success_items.append(item)

        if sleep_s: time.sleep(sleep_s)

    n = len(per_item)
    n_ok = len(success_items)
    def avg(items: List[Dict[str, Any]], k: str) -> float:
        return (sum(float(x[k]) for x in items) / len(items)) if items else 0.0

    latency_overall = {
        "mean_ms": round(mean(lat_all), 2) if lat_all else 0.0,
        "p50_ms": round(_percentile(lat_all, 50), 2) if lat_all else 0.0,
        "p95_ms": round(_percentile(lat_all, 95), 2) if lat_all else 0.0,
        "max_ms": round(max(lat_all), 2) if lat_all else 0.0,
    }
    latency_success = {
        "mean_ms": round(mean(lat_ok), 2) if lat_ok else 0.0,
        "p50_ms": round(_percentile(lat_ok, 50), 2) if lat_ok else 0.0,
        "p95_ms": round(_percentile(lat_ok, 95), 2) if lat_ok else 0.0,
        "max_ms": round(max(lat_ok), 2) if lat_ok else 0.0,
    }

    summary = {
        "n_items": n,
        "failed_requests": failed,
        "request_success_rate": ((n - failed) / n) if n else 0.0,
        "latency_ms_overall": latency_overall,
        "latency_ms_success_only": latency_success,
    }

    quality_success_only = {
        "n_success": n_ok,
        "exact_match": avg(success_items, "exact_match"),
        "contains_match": avg(success_items, "contains_match"),
        "token_f1": avg(success_items, "token_f1"),
        "retrieval_hit": avg(success_items, "retrieval_hit"),
        "context_hit_rate": avg(success_items, "context_hit_rate"),
        "mrr": mean(rr_list_ok) if rr_list_ok else 0.0,
        "hit@1": (hit1_ok / n_ok) if n_ok else 0.0,
        "hit@3": (hit3_ok / n_ok) if n_ok else 0.0,
        "hit@5": (hit5_ok / n_ok) if n_ok else 0.0,
        "answer_len_tokens_mean": mean(ans_lens_ok) if ans_lens_ok else 0.0,
        "avg_context_len_tokens_mean": mean(ctx_lens_ok) if ctx_lens_ok else 0.0,
    }

    return {"summary": summary, "quality_success_only": quality_success_only, "items": per_item}

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

    # Print a compact, presentable console summary
    console = {
        **report["summary"],
        "quality_success_only": report["quality_success_only"]
    }
    print(json.dumps(console, indent=2))

if __name__ == "__main__":
    main()
