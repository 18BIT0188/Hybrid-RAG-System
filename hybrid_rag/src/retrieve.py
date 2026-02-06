# src/retrieve.py
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ---- Optional FAISS import (Cloud-safe) ----
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

RRF_K = 60

# ---- Paths ----
FAISS_PATH = "indexes/faiss.index"
CHUNKS_META_PATH = "indexes/chunks_meta.pkl"
BM25_PATH = "indexes/bm25.pkl"

# ---- In-process caches (avoid reloading every query) ----
_INDEX = None
_CHUNKS = None
_BM25 = None
_EMB_MODEL = None


def rrf_fuse(dense_ids: List[int], sparse_ids: List[int], k: int = RRF_K) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion:
      score(d) = sum_i 1 / (k + rank_i(d))
    """
    scores: Dict[int, float] = {}

    for rank, cid in enumerate(dense_ids, start=1):
        if cid < 0:
            continue
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for rank, cid in enumerate(sparse_ids, start=1):
        if cid < 0:
            continue
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused  # [(chunk_idx, rrf_score), ...]


def _load_embedding_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def load_indexes():
    """
    Load dense (FAISS optional) and sparse (BM25 required) indexes.

    Dense retrieval is gracefully disabled if:
      - faiss is not installed, or
      - faiss index file missing/invalid.
    """
    global _INDEX, _CHUNKS, _BM25

    # Load chunks metadata (required)
    if _CHUNKS is None:
        try:
            with open(CHUNKS_META_PATH, "rb") as f:
                _CHUNKS = pickle.load(f)
        except (FileNotFoundError, EOFError) as e:
            raise RuntimeError(
                "Chunk metadata index `indexes/chunks_meta.pkl` is missing or empty. "
                "Run indexing first (`python -m src.index_dense` / `python -m src.index_sparse`)."
            ) from e

    # Load BM25 index (required)
    if _BM25 is None:
        try:
            with open(BM25_PATH, "rb") as f:
                bm = pickle.load(f)
            # bm can be {"bm25": BM25Obj} or BM25Obj directly
            _BM25 = bm["bm25"] if isinstance(bm, dict) and "bm25" in bm else bm
        except (FileNotFoundError, EOFError, KeyError, TypeError) as e:
            raise RuntimeError(
                "BM25 index `indexes/bm25.pkl` is missing or invalid. "
                "Run sparse indexing (`python -m src.index_sparse`) first."
            ) from e

    # Load FAISS index (optional)
    if _INDEX is None:
        if FAISS_AVAILABLE and os.path.exists(FAISS_PATH) and os.path.getsize(FAISS_PATH) > 0:
            try:
                _INDEX = faiss.read_index(FAISS_PATH)
            except Exception:
                _INDEX = None
        else:
            _INDEX = None

    return _INDEX, _CHUNKS, _BM25


def _safe_chunk_text(c: dict) -> str:
    return (c.get("chunk_text") or c.get("text") or "").strip()


def retrieve(query: str, top_k: int = 10):
    """
    Returns:
    {
      "dense":  [(chunk_idx, dense_score), ...],
      "sparse": [(chunk_idx, sparse_score), ...],
      "rrf":    [(chunk_idx, rrf_score), ...],
      "chunks": [chunk_meta...]
    }
    """
    if not query or not query.strip():
        return {"dense": [], "sparse": [], "rrf": [], "chunks": []}

    index, chunks, bm25 = load_indexes()

    # ---------- Dense retrieval (optional) ----------
    dense_ids: List[int] = []
    dense_scores: List[float] = []
    if index is not None:
        try:
            model = _load_embedding_model()
            q = model.encode([query], normalize_embeddings=True).astype("float32")
            scores, ids = index.search(q, top_k)

            # filter invalid ids (-1)
            for cid, sc in zip(ids[0].tolist(), scores[0].tolist()):
                if cid is not None and cid >= 0:
                    dense_ids.append(int(cid))
                    dense_scores.append(float(sc))
        except Exception:
            # If anything fails in dense path, continue with sparse only
            dense_ids, dense_scores = [], []

    # ---------- Sparse retrieval (required) ----------
    toks = query.lower().split()
    sparse_scores_arr = bm25.get_scores(toks)
    sparse_ids = np.argsort(sparse_scores_arr)[::-1][:top_k].tolist()
    sparse_scores = [float(sparse_scores_arr[i]) for i in sparse_ids]

    # ---------- Fusion ----------
    if dense_ids:
        fused = rrf_fuse(dense_ids, sparse_ids, k=RRF_K)
    else:
        # fallback: sparse-only ranking exposed as "rrf" to keep downstream compatible
        fused = [(i, float(sparse_scores_arr[i])) for i in sparse_ids]

    return {
        "dense": list(zip(dense_ids, dense_scores)),
        "sparse": list(zip(sparse_ids, sparse_scores)),
        "rrf": fused,
        "chunks": chunks,
    }
