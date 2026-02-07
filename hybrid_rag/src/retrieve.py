# src/retrieve.py
import os
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
from rank_bm25 import BM25Okapi

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

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../hybrid_rag
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
DATA_DIR = os.path.join(BASE_DIR, "data")

FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_META_PATH = os.path.join(INDEX_DIR, "chunks_meta.pkl")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

# ---- Caches ----
_INDEX = None
_CHUNKS = None
_BM25 = None
_EMB_MODEL = None


def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _looks_like_lfs_pointer(path: str) -> bool:
    try:
        if not _exists_nonempty(path):
            return False
        with open(path, "rb") as f:
            head = f.read(256)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_pickle_dump(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_corpus_rows() -> List[dict]:
    if not _exists_nonempty(CORPUS_PATH):
        raise RuntimeError(
            f"Corpus file missing: {CORPUS_PATH}. "
            "Commit data/corpus.jsonl to repo or run corpus builder."
        )
    rows = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(__import__("json").loads(line))
    return rows


def _safe_chunk_text(c: dict) -> str:
    return (c.get("chunk_text") or c.get("text") or "").strip()


def _normalize_chunks(rows: List[dict]) -> List[dict]:
    out = []
    for i, r in enumerate(rows):
        txt = _safe_chunk_text(r)
        if not txt:
            continue
        out.append({
            "chunk_id": r.get("chunk_id", f"chunk_{i}"),
            "title": r.get("title", "Unknown"),
            "url": r.get("url", ""),
            "chunk_text": txt,
        })
    return out


def _ensure_chunks() -> List[dict]:
    global _CHUNKS
    if _CHUNKS is not None:
        return _CHUNKS

    # 1) try chunks_meta.pkl
    if _exists_nonempty(CHUNKS_META_PATH) and not _looks_like_lfs_pointer(CHUNKS_META_PATH):
        try:
            obj = _safe_pickle_load(CHUNKS_META_PATH)
            if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                _CHUNKS = obj
                return _CHUNKS
        except Exception:
            pass

    # 2) fallback from corpus.jsonl
    rows = _load_corpus_rows()
    _CHUNKS = _normalize_chunks(rows)

    # best-effort repair write
    try:
        _safe_pickle_dump(CHUNKS_META_PATH, _CHUNKS)
    except Exception:
        pass

    if not _CHUNKS:
        raise RuntimeError("No valid chunks found in corpus/index.")
    return _CHUNKS


def _valid_bm25(obj: Any) -> bool:
    return hasattr(obj, "get_scores") and callable(getattr(obj, "get_scores", None))


def _build_bm25_from_chunks(chunks: List[dict]):
    tokenized = []
    for c in chunks:
        tokenized.append(_safe_chunk_text(c).lower().split())
    return BM25Okapi(tokenized)


def _ensure_bm25(chunks: List[dict]):
    global _BM25
    if _BM25 is not None:
        return _BM25

    # 1) try bm25.pkl
    if _exists_nonempty(BM25_PATH) and not _looks_like_lfs_pointer(BM25_PATH):
        try:
            raw = _safe_pickle_load(BM25_PATH)
            bm = raw["bm25"] if isinstance(raw, dict) and "bm25" in raw else raw
            if _valid_bm25(bm):
                _BM25 = bm
                return _BM25
        except Exception:
            pass  # fall through to rebuild

    # 2) rebuild from chunks in-memory (robust fallback)
    bm = _build_bm25_from_chunks(chunks)
    _BM25 = bm

    # best-effort repair write
    try:
        _safe_pickle_dump(BM25_PATH, {"bm25": bm})
    except Exception:
        pass

    return _BM25


def _load_embedding_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        if not ST_AVAILABLE:
            return None
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


def rrf_fuse(dense_ids: List[int], sparse_ids: List[int], k: int = RRF_K) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        if cid >= 0:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        if cid >= 0:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def load_indexes():
    global _INDEX

    chunks = _ensure_chunks()
    bm25 = _ensure_bm25(chunks)

    # FAISS optional
    if _INDEX is None:
        if FAISS_AVAILABLE and _exists_nonempty(FAISS_PATH) and not _looks_like_lfs_pointer(FAISS_PATH):
            try:
                _INDEX = faiss.read_index(FAISS_PATH)
            except Exception:
                _INDEX = None
        else:
            _INDEX = None

    return _INDEX, chunks, bm25


def retrieve(query: str, top_k: int = 10):
    if not query or not query.strip():
        return {"dense": [], "sparse": [], "rrf": [], "chunks": []}

    index, chunks, bm25 = load_indexes()

    # Dense (optional)
    dense_ids: List[int] = []
    dense_scores: List[float] = []

    model = _load_embedding_model()
    if (index is not None) and (model is not None):
        try:
            q = model.encode([query], normalize_embeddings=True).astype("float32")
            scores, ids = index.search(q, top_k)
            for cid, sc in zip(ids[0].tolist(), scores[0].tolist()):
                if cid is not None and cid >= 0:
                    dense_ids.append(int(cid))
                    dense_scores.append(float(sc))
        except Exception:
            dense_ids, dense_scores = [], []

    # Sparse (required, now always available via rebuild)
    toks = query.lower().split()
    sparse_scores_arr = bm25.get_scores(toks)
    sparse_ids = np.argsort(sparse_scores_arr)[::-1][:top_k].tolist()
    sparse_scores = [float(sparse_scores_arr[i]) for i in sparse_ids]

    # Fusion
    if dense_ids:
        fused = rrf_fuse(dense_ids, sparse_ids, k=RRF_K)
    else:
        fused = [(i, float(sparse_scores_arr[i])) for i in sparse_ids]

    return {
        "dense": list(zip(dense_ids, dense_scores)),
        "sparse": list(zip(sparse_ids, sparse_scores)),
        "rrf": fused,
        "chunks": chunks,
    }
