import os, pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE = os.path.dirname(os.path.dirname(__file__))
CHUNKS_CSV = os.path.join(BASE, "datasets", "sec_chunks.csv")
ART = os.path.join(BASE, "artifacts")
INDEX_PATH = os.path.join(ART, "sec_faiss.index")
CHUNKS_PKL = os.path.join(ART, "chunks.pkl")

def build_faiss(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    os.makedirs(ART, exist_ok=True)
    df = pd.read_csv(CHUNKS_CSV)
    texts = df["chunk_text"].tolist()

    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump({"chunks": texts,
                     "meta": df[["chunk_id","doc_id","company","year"]].to_dict("records")}, f)

    print(f"Saved {len(texts)} vectors -> {INDEX_PATH}")
    print(f"Saved meta -> {CHUNKS_PKL}")

if __name__ == "__main__":
    build_faiss()
