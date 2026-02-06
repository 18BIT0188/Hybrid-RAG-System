# src/index_dense.py
import os
import json
import pickle
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_corpus_jsonl(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_faiss(
    corpus_path="data/corpus.jsonl",
    out_index_path="indexes/faiss.index",
    out_meta_path="indexes/chunks_meta.pkl",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64
):
    os.makedirs("indexes", exist_ok=True)

    chunks = load_corpus_jsonl(corpus_path)
    if not chunks:
        raise ValueError(f"No chunks found in {corpus_path}")

    # Support either key: chunk_text OR text
    text_key = "chunk_text" if "chunk_text" in chunks[0] else "text"
    if text_key not in chunks[0]:
        raise KeyError("Neither 'chunk_text' nor 'text' found in corpus records.")

    texts = [c[text_key] for c in chunks]

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(emb)

    faiss.write_index(index, out_index_path)
    with open(out_meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index: {out_index_path}")
    print(f"Saved metadata: {out_meta_path}")
    print(f"Indexed chunks: {len(chunks)} | dim: {dim} | text_key: {text_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--out-index", default="indexes/faiss.index")
    parser.add_argument("--out-meta", default="indexes/chunks_meta.pkl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_faiss(
        corpus_path=args.corpus,
        out_index_path=args.out_index,
        out_meta_path=args.out_meta,
        model_name=args.model,
        batch_size=args.batch_size,
    )
