
import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs.append(model.encode(batch, convert_to_numpy=True, normalize_embeddings=True))
    return np.vstack(vecs)


def load_chunks_csv(path: str) -> List[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expected columns from your pipeline: company,year,chunk_id,source_doc,text
            rows.append({
                "company": r.get("company") or r.get("ticker") or "",
                "year": int(r.get("year", 0)),
                "chunk_id": int(r.get("chunk_id", 0)),
                "source_doc": r.get("source_doc") or r.get("doc") or "",
                "text": r.get("text") or r.get("chunk") or "",
            })
    return rows


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
          id BIGSERIAL PRIMARY KEY,
          company   TEXT NOT NULL,
          year      INT  NOT NULL,
          chunk_id  INT  NOT NULL,
          source_doc TEXT NOT NULL,
          text      TEXT NOT NULL,
          embedding VECTOR(384)
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_meta ON chunks(company, year);")
    conn.commit()


def insert_chunks(conn, rows: List[dict], vectors: np.ndarray, table: str):
    assert len(rows) == len(vectors)
    # pgvector literal expects "[v1, v2, ...]"
    def vec_literal(v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"

    with conn.cursor() as cur:
        # optional: clear same company/year before reloading to avoid dupes
        companies = sorted({r["company"] for r in rows})
        years = sorted({r["year"] for r in rows})
        cur.execute(f"DELETE FROM {table} WHERE company = ANY(%s) AND year = ANY(%s);",
                    (companies, years))

        # insert
        sql = f"""
        INSERT INTO {table} (company, year, chunk_id, source_doc, text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        """
        for r, v in zip(rows, vectors):
            cur.execute(sql, (
                r["company"], r["year"], r["chunk_id"], r["source_doc"], r["text"], vec_literal(v)
            ))
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default=os.environ.get("PG_CONN", "postgresql://intel:intel@localhost:5432/intelrag"))
    ap.add_argument("--table", default="chunks")
    ap.add_argument("--csv", default="datasets/sec_chunks.csv", help="Path to chunks CSV (output of your ingestion)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=os.environ.get("EMBED_DEVICE", "mps"))
    args = ap.parse_args()

    rows = load_chunks_csv(args.csv)
    if not rows:
        print(f"No rows found in {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} chunks from {args.csv}")

    print(f"Loading embedder: {args.model} on {args.device}")
    model = SentenceTransformer(args.model, device=args.device)

    texts = [r["text"] for r in rows]
    vecs = embed_texts(model, texts)
    print(f"Encoded {len(vecs)} vectors with dim={vecs.shape[1]}")

    print(f"Connecting to {args.conn}")
    with psycopg.connect(args.conn) as conn:
        ensure_schema(conn)
        insert_chunks(conn, rows, vecs, args.table)

    print("Done. Consider building the ANN index once (optional):")
    print("  CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    print("  ANALYZE chunks;")

if __name__ == "__main__":
    main()
