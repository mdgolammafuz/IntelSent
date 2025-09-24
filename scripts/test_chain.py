
from __future__ import annotations

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import json
from rag.chain import load_chain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--q", "--question", dest="q", required=True, help="User question")
    p.add_argument("--company", type=str, default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--top-k", dest="top_k", type=int, default=None)
    p.add_argument("--config", type=str, default="config/app.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    chain = load_chain(args.config)

    # tolerate older chain objects that may not expose .run() explicitly
    if hasattr(chain, "run"):
        out = chain.run(args.q, company=args.company, year=args.year, top_k=args.top_k)
    elif callable(chain):
        out = chain(args.q, company=args.company, year=args.year, top_k=args.top_k)
    else:
        raise RuntimeError("Loaded chain has neither .run() nor is callable.")

    print("\n=== ANSWER ===")
    print(out["answer"])

    print("\n=== CONTEXTS (top) ===")
    for i, c in enumerate(out["contexts"], 1):
        snippet = c[:300].replace("\n", " ")
        ellipsis = "..." if len(c) > 300 else ""
        print(f"[{i}] {snippet}{ellipsis}")

    print("\n=== META ===")
    print(json.dumps(out.get("meta", {}), indent=2))


if __name__ == "__main__":
    main()
