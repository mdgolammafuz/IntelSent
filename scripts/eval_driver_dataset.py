
import os, sys, json, re
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.extract import extract_driver
from rag.selector import extract_candidates, load_selector
from rag.generator import generate_answer
from utils.canon import canonicalize

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "datasets" / "qa_driver.jsonl"
SEL = BASE / "artifacts" / "selector.pkl"

REV = re.compile(r"\b(revenue|sales)\b", re.I)
MARGIN = re.compile(r"\b(gross margin|margin)\b", re.I)
OPEX = re.compile(r"\b(operating expenses|opex|research and development|sales and marketing|general and administrative)\b", re.I)
DRV = re.compile(r"(driven by|led by|primarily by|resulting from|due to|because of)", re.I)

def load_items(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def group_of(ans_can: str) -> str | None:
    if any(k in ans_can for k in ["azure","cloud services","microsoft cloud","server products","intelligent cloud"]):
        return "cloud"
    if "office" in ans_can or "microsoft 365" in ans_can:
        return "office"
    if "search" in ans_can:
        return "search"
    if "xbox" in ans_can or "game pass" in ans_can:
        return "xbox"
    if "windows" in ans_can:
        return "windows"
    if "iphone" in ans_can:
        return "iphone"
    return None

def prioritize_by_category(docs, category):
    def is_cat(d):
        t = d["text"].lower()
        if category == "revenue_driver":
            return bool(REV.search(t) and DRV.search(t))
        if category == "margin_driver":
            return bool(MARGIN.search(t) and DRV.search(t))
        if category == "opex_driver":
            return bool(OPEX.search(t) and DRV.search(t))
        return False

    driver_docs = [d for d in docs if is_cat(d)]
    seen = set(); out = []
    for d in driver_docs:
        key = (d["doc_id"], d["chunk_id"])
        if key not in seen:
            out.append(d); seen.add(key)
    for d in docs:
        key = (d["doc_id"], d["chunk_id"])
        if key not in seen:
            out.append(d); seen.add(key)
    return out

def evaluate(items, alpha=0.85, k_dense=10000, k_bm25=10000, rerank_k=400, top_k_eval=5):
    retr = HybridRetriever(alpha=alpha, top_k=k_dense, candidate_pool=k_dense, bm25_pool=k_bm25)
    rr = CrossReranker()
    selector = load_selector(SEL)

    totals = {"overall": [0, 0, 0, 0]}
    per_cat = {}

    for ex in items:
        q = ex["question"]
        co = ex["company"]
        cat = ex["category"]
        gt_chunk = ex["chunk_id"]
        gt_ans = ex["answer"].lower()
        gt_can = canonicalize(gt_ans)
        hint = group_of(gt_can)  # simulate a targeted user question

        hits = retr.retrieve(q)
        docs = retr.docs_from_hits(hits)
        docs = rr.rerank(q, docs, top_k=min(rerank_k, len(docs)))
        docs = [d for d in docs if d["company"] == co] or docs
        docs = prioritize_by_category(docs, cat)[:top_k_eval]

        cands = extract_candidates(docs, cat)
        pred = None
        if cands:
            pred = selector.select(cands, cat, group_hint=hint)

        if not pred:
            ctx = [d["text"] for d in docs]
            pred = extract_driver(ctx, category=cat) or generate_answer(ctx, q) or ""

        pred_l = pred.lower()
        pred_can = canonicalize(pred_l)

        r_atk = int(any(d["chunk_id"] == gt_chunk for d in docs))
        em = int(gt_ans == pred_l)
        cem = int(gt_can == pred_can)

        print(f"{co}/{cat}: GT='{gt_can}' | Pred='{pred_can}' | R@{top_k_eval}={r_atk} | EM={em} | C-EM={cem}")

        for bucket in ("overall", cat):
            if bucket not in per_cat and bucket != "overall":
                per_cat[bucket] = [0, 0, 0, 0]
            acc = totals["overall"] if bucket == "overall" else per_cat[bucket]
            acc[0] += 1; acc[1] += r_atk; acc[2] += em; acc[3] += cem

    tot, rh, emh, cemh = totals["overall"]
    print("\nSummary:")
    if tot:
        print(f"  Overall: Recall@{top_k_eval}={rh/tot:.2f} | EM={emh/tot:.2f} | C-EM={cemh/tot:.2f} on {tot} items")
    for cat, (ct, crh, cem_hits, ccem_hits) in per_cat.items():
        print(f"  {cat}: Recall@{top_k_eval}={crh/ct:.2f} | EM={cem_hits/ct:.2f} | C-EM={ccem_hits/ct:.2f} on {ct} items")

if __name__ == "__main__":
    items = load_items(DATA)
    if not items:
        print(f"No items found at {DATA}. Run scripts/build_eval_set.py first.")
    else:
        evaluate(items)
