import time
import os
import sys
import subprocess

import streamlit as st

# ----------------------------
# Path setup
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------
# Helpers
# ----------------------------
def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def ensure_indexes() -> tuple[bool, str]:
    """
    Ensure required retrieval artifacts exist.
    Required: indexes/chunks_meta.pkl, indexes/bm25.pkl
    Optional: indexes/faiss.index
    """
    required = [
        os.path.join(PROJECT_ROOT, "indexes", "chunks_meta.pkl"),
        os.path.join(PROJECT_ROOT, "indexes", "bm25.pkl"),
    ]

    missing = [p for p in required if not _exists_nonempty(p)]
    if not missing:
        return True, "Indexes found."

    # Try auto-build sparse index (requires corpus prepared)
    try:
        # index_sparse should create bm25 + chunks_meta in your project flow
        # if your project needs index_dense first, add that command too
        subprocess.run(
            [sys.executable, "-m", "src.index_sparse"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() if e.stderr else str(e)
        return False, (
            "Indexes are missing and auto-build failed.\n\n"
            f"Missing:\n- " + "\n- ".join(missing) + "\n\n"
            f"Build error:\n{msg}"
        )

    # re-check
    missing_after = [p for p in required if not _exists_nonempty(p)]
    if missing_after:
        return False, (
            "Auto-build ran, but required files are still missing:\n- "
            + "\n- ".join(missing_after)
        )

    return True, "Indexes built successfully."

@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Import after path setup; cached once
    from src.retrieve import retrieve
    from src.generate import answer
    return retrieve, answer

def get_chunk_text(c: dict) -> str:
    return (c.get("chunk_text") or c.get("text") or "").strip()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Hybrid RAG", page_icon="🔎", layout="wide")
st.title("Hybrid RAG (Dense + BM25 + RRF)")

ok, msg = ensure_indexes()
if not ok:
    st.error(msg)
    st.stop()
else:
    st.caption(msg)

retrieve, answer = load_pipeline()

q = st.text_input("Enter your question:")
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Top-K per retriever", 3, 30, 10)
with col2:
    top_n = st.slider("Top-N fused chunks", 1, 10, 5)

if st.button("Ask", type="primary") and q.strip():
    t0 = time.time()
    try:
        ret = retrieve(q, top_k=top_k)
        ans, top = answer(q, ret, top_n=top_n)
    except Exception as e:
        st.error(f"Error while querying RAG pipeline: {e}")
        st.stop()
    t1 = time.time()

    st.subheader("Answer")
    st.success(ans)
    st.caption(f"Response time: {t1 - t0:.3f}s")

    # quick retrieval diagnostics
    dense_count = len(ret.get("dense", []))
    sparse_count = len(ret.get("sparse", []))
    rrf_count = len(ret.get("rrf", []))
    st.write(f"Retrieved — Dense: **{dense_count}**, Sparse: **{sparse_count}**, RRF: **{rrf_count}**")

    st.subheader("Top Retrieved Chunks (RRF)")
    chunks = ret.get("chunks", [])
    if not top:
        st.warning("No chunks returned.")
    else:
        for rank, (idx, rrf_score) in enumerate(top, start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            c = chunks[idx]
            title = c.get("title", "Unknown Title")
            url = c.get("url", "")
            text = get_chunk_text(c)
            preview = (text[:800] + "...") if len(text) > 800 else text

            st.markdown(
                f"**#{rank} | RRF={float(rrf_score):.6f}**  \n"
                f"**{title}**  \n{url}"
            )
            st.write(preview if preview else "_No text available for this chunk._")
            st.divider()
