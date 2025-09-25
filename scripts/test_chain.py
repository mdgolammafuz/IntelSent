
from __future__ import annotations

import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import Any, Dict, List

from rag.chain import load_chain


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick test runner for the RAG chain")
    p.add_argument("--q", required=True, help="Question to ask")
    p.add_argument("--company", default=None, help="Company ticker, e.g. MSFT")
    p.add_argument("--year", type=int, default=None, help="Filing year, e.g. 2022")
    p.add_argument("--top-k", type=int, default=5, help="Top-k chunks")
    p.add_argument("--config", default="config/app.yaml", help="Path to YAML config")
    p.add_argument(
        "--no-openai",
        action="store_true",
        help="Force local (no-LLM) answer even if OPENAI_API_KEY is set",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Create chain from config
    chain = load_chain(args.config)

    # If the user passed --no-openai, force local answering path
    if args.no_openai:
        chain.use_openai = False

    # Run the query
    result: Dict[str, Any] = chain.run(
        question=args.q,
        company=args.company,
        year=args.year,
        top_k=args.top_k,
    )

    # Pretty-print
    print("\n=== ANSWER ===")
    print(result.get("answer", ""))

    contexts: List[str] = result.get("contexts", [])
    print("\n=== CONTEXTS (top) ===")
    for i, c in enumerate(contexts[:5], start=1):
        # Precompute short to avoid f-string backslash pitfalls on some shells
        short = c[:300].replace("\n", " ")
        suffix = "..." if len(c) > 300 else ""
        print(f"[{i}] {short}{suffix}")

    print("\n=== META ===")
    print(result.get("meta", {}))


if __name__ == "__main__":
    main()
