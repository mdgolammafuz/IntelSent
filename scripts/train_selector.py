import os, sys, json, re
from pathlib import Path
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from joblib import dump
from sklearn.linear_model import LogisticRegression
import numpy as np

from rag.retriever import HybridRetriever
from rag.rerank import CrossReranker
from rag.selector import extract_candidates, load_selector
from utils.canon import canonicalize

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "datasets" / "qa_driver.jsonl"
OUT = BASE / "artifacts" / "selector.pkl"

def load_items(path: Path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def features_for(cand, category):
    # Must mirror rag/selector.py feature_names order
    feats = {
        "bias": 1.0,
        "ctx_rank": cand.ctx_rank,
        "ctx_rank_inv": 1.0 / (1 + cand.ctx_rank),
        "ce_score": cand.ce_score,
        "len_words": len(cand.phrase.split()),
        "subj_any": 1.0 if cand.subj_groups else 0.0,
        "g_search": 1.0 if "search" in cand.cand_groups else 0.0,
        "g_office": 1.0 if "office" in cand.cand_groups else 0.0,
        "g_xbox": 1.0 if "xbox" in cand.cand_groups else 0.0,
        "g_windows": 1.0 if "windows" in cand.cand_groups else 0.0,
        "g_cloud": 1.0 if "cloud" in cand.cand_groups else 0.0,
        "g_iphone": 1.0 if "iphone" in cand.cand_groups else 0.0,
        "align": 1.0 if any(g in cand.cand_groups for g in cand.subj_groups) else 0.0,
        "cat_rev": 1.0 if category == "revenue_driver" else 0.0,
        "cat_margin": 1.0 if category == "margin_driver" else 0.0,
        "cat_opex": 1.0 if category == "opex_driver" else 0.0,
    }
    return feats

def evaluate_once(model, feat_names, X, y):
    if not len(y):
        return 0.0
    probs = model.predict_proba(np.array(X))[:, 1]
    preds = (probs >= 0.5).astype(int)
    return float((preds == np.array(y)).mean())

def main():
    items = load_items(DATA)
    retr = HybridRetriever(alpha=0.85, top_k=10000, candidate_pool=10000, bm25_pool=10000)
    rr = CrossReranker()

    feat_names = [
        "bias","ctx_rank","ctx_rank_inv","ce_score","len_words","subj_any",
        "g_search","g_office","g_xbox","g_windows","g_cloud","g_iphone",
        "align","cat_rev","cat_margin","cat_opex"
    ]
    X, y = [], []

    kept = 0
    for ex in items:
        q, co, cat = ex["question"], ex["company"], ex["category"]
        gt = canonicalize(ex["answer"])

        hits = retr.retrieve(q)
        docs = retr.docs_from_hits(hits)
        docs = rr.rerank(q, docs, top_k=min(400, len(docs)))
        docs = [d for d in docs if d["company"] == co] or docs

        # prioritize by category then cut
        def is_cat(d):
            t = d["text"].lower()
            if cat == "revenue_driver":
                return ("revenue" in t or "sales" in t) and ("driven by" in t or "led by" in t or "due to" in t)
            if cat == "margin_driver":
                return "margin" in t and ("driven by" in t or "led by" in t or "due to" in t)
            if cat == "opex_driver":
                return ("operating expenses" in t or "r&d" in t or "sales and marketing" in t or "general and administrative" in t) and ("driven by" in t or "led by" in t or "due to" in t)
            return True
        prio = [d for d in docs if is_cat(d)]
        seen = set(); ordered=[]
        for d in prio + docs:
            key = (d["doc_id"], d["chunk_id"])
            if key not in seen:
                ordered.append(d); seen.add(key)
        docs = ordered[:5]

        cands = extract_candidates(docs, cat)
        if not cands:
            continue

        # Label: canonical match with ground truth
        labels = [1 if canonicalize(c.phrase) == gt else 0 for c in cands]
        if not any(labels):
            continue  # skip items with no positive candidate in top-5

        for c, lab in zip(cands, labels):
            feats = features_for(c, cat)
            X.append([feats[n] for n in feat_names])
            y.append(lab)
        kept += 1

    if not X:
        print("No training data found. Did candidate extraction fail?")
        return

    X = np.array(X); y = np.array(y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "feature_names": feat_names}, OUT)
    print(f"Saved selector -> {OUT} from {kept} items")

if __name__ == "__main__":
    main()
