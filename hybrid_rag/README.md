## hybrid_rag

Hybrid Retrieval-Augmented Generation (RAG) system using:

- **Dense retrieval** (sentence embeddings + FAISS)
- **Sparse retrieval** (BM25)
- **Reciprocal Rank Fusion (RRF)**
- **LLM-based answer generation**
- **Automated evaluation** (100 generated questions + metrics + PDF report)

---

### Structure

- `data/`: URL lists + cleaned corpus chunks (`corpus.jsonl`)
- `indexes/`: dense (FAISS) + sparse (BM25) artifacts + metadata
- `eval/`: questions, run outputs, metrics, and PDF report
- `app/`: Streamlit UI
- `src/`: collection, preprocessing, chunking, indexing, retrieval, generation, evaluation, report, and pipeline scripts

---

## Installation

```bash
cd hybrid_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
