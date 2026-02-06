# src/index_sparse.py
import os
import json
import pickle
import argparse
from rank_bm25 import BM25Okapi


def load_corpus_jsonl(path: str):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_bm25(corpus_path="data/corpus.jsonl", out_path="indexes/bm25.pkl"):
    os.makedirs("indexes", exist_ok=True)

    chunks = load_corpus_jsonl(corpus_path)
    if not chunks:
        raise ValueError(f"No chunks found in {corpus_path}")

    text_key = "chunk_text" if "chunk_text" in chunks[0] else "text"
    if text_key not in chunks[0]:
        raise KeyError("Neither 'chunk_text' nor 'text' found in corpus records.")

    tokenized = [c[text_key].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks, "text_key": text_key}, f)

    print(f"Saved BM25 index: {out_path}")
    print(f"Indexed chunks: {len(chunks)} | text_key: {text_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--out", default="indexes/bm25.pkl")
    args = parser.parse_args()

    build_bm25(corpus_path=args.corpus, out_path=args.out)
