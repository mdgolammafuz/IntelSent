
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from rag.chain import load_chain

# Silence HF tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True)
    p.add_argument("--company", default=None)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5, dest="top_k")
    p.add_argument("--config", default="config/app.yaml")
    p.add_argument(
        "--no-openai",
        action="store_true",
        help="Force local extractive mode (no OpenAI)."
    )
    args = p.parse_args()

    # Force no-OpenAI mode if flag is set
    use_lcel = not args.no_openai

    chain = load_chain(args.config, use_lcel=use_lcel)

    result = chain.run(args.q, company=args.company, year=args.year, top_k=args.top_k)

    print("\n=== ANSWER ===")
    print(result.get("answer", "").strip())

    ctxs = result.get("contexts", []) or []
    if ctxs:
        print("\n=== CONTEXTS (top) ===")
        for i, c in enumerate(ctxs[: args.top_k], 1):
            short = (c[:300] + "...") if len(c) > 300 else c
            short_one_line = short.replace("\n", " ")
            print(f"[{i}] {short_one_line}")

    meta = result.get("meta", {})
    if meta:
        print("\n=== META ===")
        print(meta)


if __name__ == "__main__":
    main()
