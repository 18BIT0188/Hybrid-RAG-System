import time
import os
import sys

import streamlit as st

# Ensure project root (which contains `src/`) is on sys.path when run via Streamlit
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieve import retrieve
from src.generate import answer

st.title("Hybrid RAG (Dense + BM25 + RRF)")

q = st.text_input("Enter your question:")
top_k = st.slider("Top-K per retriever", 5, 30, 10)
top_n = st.slider("Top-N fused chunks", 2, 10, 5)

if st.button("Ask") and q.strip():
    t0 = time.time()
    try:
        ret = retrieve(q, top_k=top_k)
        ans, top = answer(q, ret, top_n=top_n)
    except RuntimeError as e:
        # Typically raised when indexes are missing / not yet built
        st.error(str(e))
        st.stop()
    t1 = time.time()

    st.subheader("Answer")
    st.success(ans)
    st.write(ans)
    st.write(f"Response time: {t1 - t0:.3f}s")

    st.subheader("Top Retrieved Chunks (RRF)")
    for rank, (idx, rrf_score) in enumerate(top, start=1):
        c = ret["chunks"][idx]
        st.markdown(f"**#{rank} | RRF={rrf_score:.6f}**  \n**{c['title']}**  \n{c['url']}")
        st.write(c["chunk_text"][:800] + "...")
        st.divider()
