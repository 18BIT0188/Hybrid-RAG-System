"""
End-to-end pipeline helpers to go from URLs -> corpus -> indexes.
"""

from .preprocess import build_corpus
from .index_dense import _load_corpus as _load_corpus_for_dense, build_faiss
from .index_sparse import _load_corpus as _load_corpus_for_sparse, build_bm25


def run_all() -> None:
    """
    1) Build `data/corpus.jsonl` from `data/fixed_urls.json` + `data/random_urls.json`
    2) Build dense FAISS index + chunk metadata
    3) Build BM25 sparse index
    """
    # Step 1: corpus
    build_corpus()

    # Step 2: dense
    corpus_for_dense = _load_corpus_for_dense()
    build_faiss(corpus_for_dense)

    # Step 3: sparse
    corpus_for_sparse = _load_corpus_for_sparse()
    build_bm25(corpus_for_sparse)


if __name__ == "__main__":
    run_all()

