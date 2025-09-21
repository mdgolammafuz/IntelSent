import os, sys, re, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PKL = BASE / "artifacts" / "chunks.pkl"

with open(PKL, "rb") as f:
    data = pickle.load(f)

needle = re.compile(r"driven by azure", re.I)
hits = [(i, m, t) for i, (t, m) in enumerate(zip(data["chunks"], data["meta"])) if needle.search(t)]
for i, m, t in hits[:5]:
    print(f"chunk_id={m['chunk_id']} doc_id={m['doc_id']} company={m['company']} year={m['year']}")
    print(t.strip()[:300], "\n")
print(f"total hits: {len(hits)}")
