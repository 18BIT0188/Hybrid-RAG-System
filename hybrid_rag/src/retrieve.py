# src/retrieve.py
import os
import sys
import pickle
import subprocess
from typing import Dict, List, Tuple, Any

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


def _looks_like_lfs_pointer(path: str) -> bool:
    """
    Detect git-lfs pointer text file instead of real binary.
    """
    if not _exists_nonempty(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _is_valid_chunks_obj(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))


def _is_valid_bm25_obj(obj: Any) -> bool:
    # bm25 object from rank_bm25 exposes get_scores
    return hasattr(obj, "get_scores") and callable(getattr(obj, "get_scores", None))


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _attempt_rebuild_indexes() -> Tuple[bool, str]:
    """
    Try to rebuild sparse indexes in-place.
    """
    try:
        # This should regenerate bm25.pkl (+ usually chunks_meta.pkl)
        proc = subprocess.run(
            [sys.executable, "-m", "src.index_sparse"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, (proc.stdout or "").strip()
    except Exception as e:
        return False, str(e)


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


def _load_chunks_meta_with_repair():
    global _CHUNKS

    if _CHUNKS is not None:
        return _CHUNKS

    bad_reason = None
    needs_rebuild = False

    if not _exists_nonempty(CHUNKS_META_PATH):
        bad_reason = "missing/empty"
        needs_rebuild = True
    elif _looks_like_lfs_pointer(CHUNKS_META_PATH):
        bad_reason = "git-lfs pointer file"
        needs_rebuild = True

    if needs_rebuild:
        ok, out = _attempt_rebuild_indexes()
        if not ok:
            raise RuntimeError(
                f"Chunk metadata index `{CHUNKS_META_PATH}` is {bad_reason}. "
                f"Auto-rebuild failed: {out}"
            )

    try:
        obj = _safe_pickle_load(CHUNKS_META_PATH)
        if not _is_valid_chunks_obj(obj):
            raise ValueError("invalid chunks structure")
        _CHUNKS = obj
        return _CHUNKS
    except Exception:
        # one more rebuild attempt if corrupted
        ok, out = _attempt_rebuild_indexes()
        if ok:
            try:
                obj = _safe_pickle_load(CHUNKS_META_PATH)
                if not _is_valid_chunks_obj(obj):
                    raise ValueError("invalid chunks structure after rebuild")
                _CHUNKS = obj
                return _CHUNKS
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load chunk metadata from `{CHUNKS_META_PATH}` after rebuild."
                ) from e2

        raise RuntimeError(
            f"Failed to load chunk metadata from `{CHUNKS_META_PATH}`. File may be corrupt."
        )


def _load_bm25_with_repair():
    global _BM25

    if _BM25 is not None:
        return _BM25

    bad_reason = None
    needs_rebuild = False

    if not _exists_nonempty(BM25_PATH):
        bad_reason = "missing/empty"
        needs_rebuild = True
    elif _looks_like_lfs_pointer(BM25_PATH):
        bad_reason = "git-lfs pointer file"
        needs_rebuild = True

    if needs_rebuild:
        ok, out = _attempt_rebuild_indexes()
        if not ok:
            raise RuntimeError(
                f"BM25 index `{BM25_PATH}` is {bad_reason}. Auto-rebuild failed: {out}"
            )

    def _extract_bm25(obj: Any):
        return obj["bm25"] if isinstance(obj, dict) and "bm25" in obj else obj

    try:
        obj = _safe_pickle_load(BM25_PATH)
        bm = _extract_bm25(obj)
        if not _is_valid_bm25_obj(bm):
            raise ValueError("invalid bm25 object")
        _BM25 = bm
        return _BM25
    except Exception:
        # one more rebuild attempt if corrupted/incompatible
        ok, out = _attempt_rebuild_indexes()
        if ok:
            try:
                obj = _safe_pickle_load(BM25_PATH)
                bm = _extract_bm25(obj)
                if not _is_valid_bm25_obj(bm):
                    raise ValueError("invalid bm25 object after rebuild")
                _BM25 = bm
                return _BM25
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load BM25 index from `{BM25_PATH}` after rebuild."
                ) from e2

        raise RuntimeError(
            f"Failed to load BM25 index from `{BM25_PATH}`. File may be corrupt."
        )


def load_indexes():
    """
    Load dense (FAISS optional) and sparse (BM25 required) indexes.

    Dense retrieval is gracefully disabled if:
      - faiss is not installed, or
      - sentence-transformers is not installed, or
      - faiss index file missing/invalid.
    """
    global _INDEX

    chunks = _load_chunks_meta_with_repair()
    bm25 = _load_bm25_with_repair()

    # Load FAISS index (optional)
    if _INDEX is None:
        if FAISS_AVAILABLE and _exists_nonempty(FAISS_PATH) and not _looks_like_lfs_pointer(FAISS_PATH):
            try:
                _INDEX = faiss.read_index(FAISS_PATH)
            except Exception:
                _INDEX = None
        else:
            _INDEX = None

    return _INDEX, chunks, bm25


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
