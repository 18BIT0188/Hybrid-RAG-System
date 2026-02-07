# src/retrieve.py
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

# ---- Optional sentence-transformers import (Cloud-safe) ----
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    ST_AVAILABLE = False

# ---- Optional FAISS import (Cloud-safe) ----
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

RRF_K = 60

# ---- Robust absolute paths (important for Streamlit Cloud) ----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../hybrid_rag
INDEX_DIR = os.path.join(BASE_DIR, "indexes")

FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_META_PATH = os.path.join(INDEX_DIR, "chunks_meta.pkl")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")

# ---- In-process caches (avoid reloading every query) ----
_INDEX = None
_CHUNKS = None
_BM25 = None
_EMB_MODEL = None


def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


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
        if not ST_AVAILABLE:
            return None
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def load_indexes():
    """
    Load dense (FAISS optional) and sparse (BM25 required) indexes.

    Dense retrieval is gracefully disabled if:
      - faiss is not installed, or
      - sentence-transformers is not installed, or
      - faiss index file missing/invalid.
    """
    global _INDEX, _CHUNKS, _BM25

    # Load chunks metadata (required)
    if _CHUNKS is None:
        if not _exists_nonempty(CHUNKS_META_PATH):
            raise RuntimeError(
                f"Chunk metadata index `{CHUNKS_META_PATH}` is missing or empty. "
                "Run indexing first (`python -m src.index_dense` / `python -m src.index_sparse`) "
                "and ensure the generated files are available in deployment."
            )
        try:
            with open(CHUNKS_META_PATH, "rb") as f:
                _CHUNKS = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load chunk metadata from `{CHUNKS_META_PATH}`. File may be corrupt."
            ) from e

    # Load BM25 index (required)
    if _BM25 is None:
        if not _exists_nonempty(BM25_PATH):
            raise RuntimeError(
                f"BM25 index `{BM25_PATH}` is missing or empty. "
                "Run sparse indexing (`python -m src.index_sparse`) and deploy index files."
            )
        try:
            with open(BM25_PATH, "rb") as f:
                bm = pickle.load(f)
            _BM25 = bm["bm25"] if isinstance(bm, dict) and "bm25" in bm else bm
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BM25 index from `{BM25_PATH}`. File may be corrupt."
            ) from e

    # Load FAISS index (optional)
    if _INDEX is None:
        if FAISS_AVAILABLE and _exists_nonempty(FAISS_PATH):
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

    model = _load_embedding_model()
    dense_enabled = (index is not None) and (model is not None)

    if dense_enabled:
        try:
            q = model.encode([query], normalize_embeddings=True).astype("float32")
            scores, ids = index.search(q, top_k)

            # filter invalid ids (-1)
            for cid, sc in zip(ids[0].tolist(), scores[0].tolist()):
                if cid is not None and cid >= 0:
                    dense_ids.append(int(cid))
                    dense_scores.append(float(sc))
        except Exception:
            # continue with sparse-only
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
