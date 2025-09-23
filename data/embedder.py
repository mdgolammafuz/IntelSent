
# Build FAISS index from datasets/sec_chunks.csv
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# sentence-transformers
from sentence_transformers import SentenceTransformer
import faiss
import torch

BASE = Path(__file__).resolve().parents[1]
CHUNKS_CSV = BASE / "datasets" / "sec_chunks.csv"
ART_DIR = BASE / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = ART_DIR / "sec_faiss.index"
META_PATH = ART_DIR / "chunks.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_texts() -> pd.DataFrame:
    if not CHUNKS_CSV.exists():
        raise FileNotFoundError(f"{CHUNKS_CSV} not found. Run data/chunker.py first.")
    df = pd.read_csv(CHUNKS_CSV)
    required = {"chunk_id", "doc_id", "company", "year", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{CHUNKS_CSV} missing columns: {missing}")
    df["chunk_id"] = df["chunk_id"].astype(int)
    df["year"] = df["year"].astype(int)
    df["text"] = df["text"].astype(str)
    return df

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
        batch = texts[i:i+BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.append(np.asarray(vecs, dtype="float32"))
    return np.vstack(embs)

def build_faiss():
    df = load_texts()
    texts = df["text"].tolist()

    device = pick_device()
    print(f"Embedding on device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    mat = embed_texts(model, texts)  # (N, D), L2-normalized

    d = mat.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine (with normalized vectors)
    index.add(mat)
    faiss.write_index(index, str(INDEX_PATH))

    meta: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        meta.append({
            "chunk_id": int(r["chunk_id"]),
            "doc_id": r["doc_id"],
            "company": r["company"],
            "year": int(r["year"]),
            "text": r["text"],
        })
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved {len(df)} vectors -> {INDEX_PATH}")
    print(f"Saved meta -> {META_PATH}")

if __name__ == "__main__":
    build_faiss()
