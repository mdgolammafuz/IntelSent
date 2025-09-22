import os, csv
import pandas as pd
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.dirname(__file__))
DOCS_CSV = os.path.join(BASE, "datasets", "sec", "sec_docs.csv")
CHUNKS_CSV = os.path.join(BASE, "datasets", "sec_chunks.csv")

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_count(text: str) -> list[int]:
    # return token ids so we can cut by tokens, not characters
    return tok.encode(text, add_special_tokens=False)

def chunk_text(text: str, max_tokens=256, overlap=32):
    ids = tokenize_count(text)
    n = len(ids)
    i = 0
    while i < n:
        sub_ids = ids[i:i+max_tokens]
        # decode back to text for storage
        yield tok.decode(sub_ids, clean_up_tokenization_spaces=True)
        if i + max_tokens >= n: break
        i += max_tokens - overlap

def main():
    df = pd.read_csv(DOCS_CSV)
    out_rows = []
    chunk_id = 0
    for _, r in df.iterrows():
        for ch in chunk_text(r["text"]):
            out_rows.append({
                "chunk_id": chunk_id,
                "doc_id": r["doc_id"],
                "company": r["company"],
                "year": int(r["year"]),
                "chunk_text": ch
            })
            chunk_id += 1
    pd.DataFrame(out_rows).to_csv(CHUNKS_CSV, index=False)
    print(f"Wrote {len(out_rows)} chunks to {CHUNKS_CSV}")

if __name__ == "__main__":
    main()
