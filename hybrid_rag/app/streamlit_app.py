import os
import sys
import time
from typing import Dict, List, Tuple

import streamlit as st

# -------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so `from src...` works
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports
from src.retrieve import retrieve  # noqa: E402
from src.generate import answer     # noqa: E402


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _safe_get_chunk(chunks: List[dict], idx: int) -> dict:
    if 0 <= idx < len(chunks):
        return chunks[idx]
    return {
        "title": "Unknown",
        "url": "N/A",
        "chunk_text": "",
        "chunk_id": f"missing::{idx}",
    }


def _build_rank_map(results: List[Tuple[int, float]]) -> Dict[int, int]:
    """
    Build a map: chunk_idx -> rank (1-based)
    """
    rank_map = {}
    for r, (chunk_idx, _score) in enumerate(results, start=1):
        rank_map[chunk_idx] = r
    return rank_map


def _compact_text(text: str, max_chars: int = 900) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_sec(seconds: float) -> str:
    ms = seconds * 1000.0
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{seconds:.2f} s"


# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid RAG (Dense + BM25 + RRF)",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Hybrid RAG System")
st.caption("Dense Retrieval + BM25 Sparse Retrieval + Reciprocal Rank Fusion (RRF)")

# Sidebar controls
st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider("Top-K per retriever", min_value=3, max_value=50, value=10, step=1)
top_n = st.sidebar.slider("Top-N chunks for generation", min_value=1, max_value=10, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### RRF Formula")
st.sidebar.latex(r"\mathrm{RRF}(d)=\sum_i \frac{1}{k+\mathrm{rank}_i(d)}")
st.sidebar.write("Using assignment default: **k = 60**")

show_dense = st.sidebar.checkbox("Show dense results", value=True)
show_sparse = st.sidebar.checkbox("Show sparse results", value=True)
show_rrf = st.sidebar.checkbox("Show fused (RRF) results", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "If you see index errors, build indexes first:\n\n"
    "`python -m src.preprocess`\n"
    "`python -m src.index_dense`\n"
    "`python -m src.index_sparse`"
)

# Query input
default_q = "Explain the process of photosynthesis in simple terms."
query = st.text_area("Enter your question", value=default_q, height=90)

ask_col, clear_col = st.columns([1, 1])
ask = ask_col.button("Ask", type="primary", use_container_width=True)
clear = clear_col.button("Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

# -------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------
if ask:
    q = (query or "").strip()
    if not q:
        st.warning("Please enter a question.")
        st.stop()

    # Timing breakdown
    t0 = time.perf_counter()

    try:
        t_retr_start = time.perf_counter()
        ret = retrieve(q, top_k=top_k)
        t_retr_end = time.perf_counter()

        t_gen_start = time.perf_counter()
        ans, top_rrf = answer(q, ret, top_n=top_n)
        t_gen_end = time.perf_counter()
    except FileNotFoundError as e:
        st.error(
            "Required index file is missing.\n\n"
            "Please run:\n"
            "1) `python -m src.preprocess`\n"
            "2) `python -m src.index_dense`\n"
            "3) `python -m src.index_sparse`\n\n"
            f"Details: {e}"
        )
        st.stop()
    except Exception as e:
        st.error(f"Error while querying RAG pipeline: {e}")
        st.stop()

    t1 = time.perf_counter()

    total_time = t1 - t0
    retr_time = t_retr_end - t_retr_start
    gen_time = t_gen_end - t_gen_start

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Time", _format_sec(total_time))
    m2.metric("Retrieval Time", _format_sec(retr_time))
    m3.metric("Generation Time", _format_sec(gen_time))
    m4.metric("Top-N used", str(len(top_rrf)))

    # Answer box
    st.subheader("Answer")
    st.success(ans if ans else "I don't know")

    chunks = ret.get("chunks", [])
    dense_results = ret.get("dense", [])     # list[(chunk_idx, dense_score)]
    sparse_results = ret.get("sparse", [])   # list[(chunk_idx, sparse_score)]
    rrf_results = ret.get("rrf", [])         # list[(chunk_idx, rrf_score)]

    dense_rank_map = _build_rank_map(dense_results)
    sparse_rank_map = _build_rank_map(sparse_results)

    # Top context used for generation
    st.subheader("Top Retrieved Chunks (Used for Generation)")
    if not top_rrf:
        st.info("No chunks were retrieved.")
    else:
        for rank, (chunk_idx, rrf_score) in enumerate(top_rrf, start=1):
            c = _safe_get_chunk(chunks, chunk_idx)
            d_rank = dense_rank_map.get(chunk_idx, None)
            s_rank = sparse_rank_map.get(chunk_idx, None)

            with st.expander(
                f"#{rank} | RRF={rrf_score:.6f} | {c.get('title', 'Unknown')}",
                expanded=(rank <= 2),
            ):
                st.markdown(f"**URL:** {c.get('url', 'N/A')}")
                st.markdown(f"**Chunk ID:** `{c.get('chunk_id', f'idx::{chunk_idx}')}`")
                st.markdown(
                    f"**Dense rank:** `{d_rank if d_rank is not None else '—'}`  \n"
                    f"**Sparse rank:** `{s_rank if s_rank is not None else '—'}`"
                )
                st.write(_compact_text(c.get("chunk_text", ""), max_chars=1200))

    # Optional debug/result sections
    col_a, col_b, col_c = st.columns(3)

    if show_dense:
        with col_a:
            st.subheader("Dense Top-K")
            if dense_results:
                for r, (idx, score) in enumerate(dense_results, start=1):
                    c = _safe_get_chunk(chunks, idx)
                    st.markdown(
                        f"**{r}.** score={score:.4f}  \n"
                        f"{c.get('title','Unknown')}  \n"
                        f"{c.get('url','N/A')}"
                    )
            else:
                st.caption("No dense results.")

    if show_sparse:
        with col_b:
            st.subheader("Sparse (BM25) Top-K")
            if sparse_results:
                for r, (idx, score) in enumerate(sparse_results, start=1):
                    c = _safe_get_chunk(chunks, idx)
                    st.markdown(
                        f"**{r}.** score={score:.4f}  \n"
                        f"{c.get('title','Unknown')}  \n"
                        f"{c.get('url','N/A')}"
                    )
            else:
                st.caption("No sparse results.")

    if show_rrf:
        with col_c:
            st.subheader("RRF Top-K")
            if rrf_results:
                show_n = min(len(rrf_results), max(top_k, 10))
                for r, (idx, score) in enumerate(rrf_results[:show_n], start=1):
                    c = _safe_get_chunk(chunks, idx)
                    st.markdown(
                        f"**{r}.** RRF={score:.6f}  \n"
                        f"{c.get('title','Unknown')}  \n"
                        f"{c.get('url','N/A')}"
                    )
            else:
                st.caption("No RRF results.")

    st.markdown("---")
    st.caption(
        "Tip: If answers look like prompt/context echoes, tighten generation prompt in `src/generate.py` "
        "and keep `do_sample=False`, `num_beams>=2`."
    )
