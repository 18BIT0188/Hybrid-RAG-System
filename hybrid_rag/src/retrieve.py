import os
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

RRF_K = 60


def rrf_fuse(dense_ids, sparse_ids, k=RRF_K):
    # dense_ids: list of chunk indices (ranked)
    # sparse_ids: list of chunk indices (ranked)
    scores = {}
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused  # [(chunk_idx, rrf_score), ...]


def load_indexes():
    """
    Load dense (FAISS) and sparse (BM25) indexes.

    If the FAISS index file is missing or invalid (e.g. still the empty
    placeholder we created when scaffolding the repo), we gracefully
    disable dense retrieval by returning `index=None`.
    """
    index = None
    faiss_path = "indexes/faiss.index"
    if os.path.exists(faiss_path) and os.path.getsize(faiss_path) > 0:
        try:
            index = faiss.read_index(faiss_path)
        except RuntimeError:
            # Invalid / corrupt index – treat as unavailable
            index = None

    try:
        with open("indexes/chunks_meta.pkl", "rb") as f:
            chunks = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        raise RuntimeError(
            "Chunk metadata index `indexes/chunks_meta.pkl` is missing or empty. "
            "Run your indexing pipeline (e.g. `python src/index_dense.py` and "
            "`python src/index_sparse.py`) to build the indexes before querying."
        ) from e

    try:
        with open("indexes/bm25.pkl", "rb") as f:
            bm = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        raise RuntimeError(
            "BM25 index `indexes/bm25.pkl` is missing or empty. "
            "Run your sparse indexing step (e.g. `python src/index_sparse.py`) "
            "to build the BM25 index before querying."
        ) from e
    return index, chunks, bm["bm25"]


def retrieve(query: str, top_k=10):
    index, chunks, bm25 = load_indexes()

    # Dense (optional – only if a valid FAISS index is available)
    dense_ids: list[int] = []
    dense_scores: list[float] = []
    if index is not None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(q, top_k)
        dense_ids = ids[0].tolist()
        dense_scores = scores[0].tolist()

    # Sparse
    sparse_scores = bm25.get_scores(query.lower().split())
    sparse_ids = np.argsort(sparse_scores)[::-1][:top_k].tolist()

    fused = rrf_fuse(dense_ids, sparse_ids, k=60) if dense_ids else [
        (i, float(sparse_scores[i])) for i in sparse_ids
    ]
    return {
        "dense": list(zip(dense_ids, dense_scores)),
        "sparse": [(i, float(sparse_scores[i])) for i in sparse_ids],
        "rrf": fused,
        "chunks": chunks,
    }
