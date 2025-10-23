# Embed chunks CSV and upsert into pgvector table `chunks`
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import psycopg
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims

def load_chunks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run data/chunker.py first.")
    df = pd.read_csv(path)
    req = {"chunk_id", "doc_id", "company", "year", "text"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df["year"] = df["year"].astype(int)
    return df

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    outs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        vecs = model.encode(texts[i:i+batch_size], normalize_embeddings=True, show_progress_bar=False)
        outs.append(np.asarray(vecs, dtype="float32"))
    return np.vstack(outs) if outs else np.zeros((0, 384), dtype="float32")

def ensure_table(conn: psycopg.Connection, table: str):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
          id        BIGSERIAL PRIMARY KEY,
          company   TEXT NOT NULL,
          year      INT  NOT NULL,
          chunk_id  INT  NOT NULL,
          source_doc TEXT NOT NULL,
          text      TEXT NOT NULL,
          embedding VECTOR(384)
        );
        """)
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_meta ON {table}(company, year);")
    conn.commit()

def insert_rows(conn: psycopg.Connection, table: str, rows: List[Dict[str, Any]]):
    sql = f"""
    INSERT INTO {table} (company, year, chunk_id, source_doc, text, embedding)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
    """
    with conn.cursor() as cur:
        cur.executemany(sql, [
            (r["company"], r["year"], r["chunk_id"], r["doc_id"], r["text"], r["emb_literal"])
            for r in rows
        ])
    conn.commit()

def to_pgvector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to sec_chunks.csv")
    ap.add_argument("--db-dsn", required=True, help="postgresql://user:pass@host:port/db")
    ap.add_argument("--table", default="chunks")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    df = load_chunks(Path(args.chunks))
    texts = df["text"].tolist()
    model = SentenceTransformer(MODEL_NAME, device="cpu")  # consistent inside container

    embs = embed_texts(model, texts, args.batch_size)
    if len(embs) != len(df):
        raise RuntimeError("Embedding/text length mismatch")

    with psycopg.connect(args.db_dsn) as conn:
        ensure_table(conn, args.table)

        rows: List[Dict[str, Any]] = []
        for (_, r), v in zip(df.iterrows(), embs):
            rows.append({
                "company": r["company"],
                "year": int(r["year"]),
                "chunk_id": int(r["chunk_id"]),
                "doc_id": r["doc_id"],
                "text": r["text"],
                "emb_literal": to_pgvector_literal(v),
            })
            if len(rows) >= 1000:
                insert_rows(conn, args.table, rows); rows.clear()
        if rows:
            insert_rows(conn, args.table, rows)

    print(f"Loaded {len(df)} chunks into table '{args.table}'")

if __name__ == "__main__":
    main()
