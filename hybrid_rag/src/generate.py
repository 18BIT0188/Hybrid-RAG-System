# src/generate.py
from typing import Dict, List, Tuple, Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.retrieve import retrieve as retrieve_hybrid

MODEL_NAME = "google/flan-t5-base"

# Lazy-loaded globals
_TOKENIZER = None
_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _MODEL.to(_DEVICE)
        _MODEL.eval()


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _split_sentences(text: str) -> List[str]:
    text = _normalize_whitespace(text)
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip().split()) >= 6]


def build_context(
    chunks: List[dict],
    selected: List[Tuple[int, float]],
    max_chars_per_chunk: int = 900
) -> str:
    parts = []
    for rank, (idx, _score) in enumerate(selected, start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        snippet = _normalize_whitespace(c.get("chunk_text") or c.get("text") or "")[:max_chars_per_chunk]
        parts.append(
            f"[{rank}] Title: {c.get('title','')}\n"
            f"URL: {c.get('url','')}\n"
            f"Snippet: {snippet}"
        )
    return "\n\n".join(parts)


def _clean_answer(raw: str) -> str:
    txt = _normalize_whitespace(raw)

    # Strip prompt echoes
    markers = ["Question:", "Context:", "Final answer:", "Answer:"]
    for m in markers:
        if m in txt and len(txt) > 180:
            txt = txt.split(m)[-1].strip()

    # trim overly long output
    if len(txt) > 700:
        txt = txt[:700].rsplit(" ", 1)[0].strip()

    return txt


def _is_bad_answer(ans: str, query: str) -> bool:
    a = _normalize_whitespace(ans).lower().strip(" .!?")
    q = _normalize_whitespace(query).lower().strip(" .!?")
    if not a:
        return True
    if len(a.split()) < 5:
        return True
    if a == q:
        return True
    if a in {"i don't know", "dont know", "unknown"}:
        return True
    if "process of" in a and len(a.split()) <= 8:
        return True
    return False


def _extractive_fallback(query: str, chunks: List[dict], top: List[Tuple[int, float]]) -> str:
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    best_sent = ""
    best_score = -1

    for idx, _ in top[:3]:
        if idx < 0 or idx >= len(chunks):
            continue
        txt = chunks[idx].get("chunk_text") or chunks[idx].get("text") or ""
        for s in _split_sentences(txt):
            s_tokens = set(re.findall(r"[a-z0-9]+", s.lower()))
            score = len(q_tokens & s_tokens)
            if score > best_score:
                best_score = score
                best_sent = s

    if best_sent:
        return _normalize_whitespace(best_sent)

    return "I don't know based on the provided context."


def answer(query: str, retrieved: Optional[Dict] = None, top_n: int = 5):
    """
    Compatible call patterns:
      - answer(query, retrieved, top_n=...)
      - answer(query, top_n=...)  # retrieved internally

    Returns:
      (final_answer: str, selected_chunks: List[(chunk_idx, score)])
    """
    _load_model()

    # Retrieval fallback if caller didn't pass retrieved
    if retrieved is None:
        retrieved = retrieve_hybrid(query, top_k=max(10, top_n * 3))

    chunks = retrieved.get("chunks", [])
    rrf = retrieved.get("rrf", [])

    if not chunks or not rrf:
        return "I don't know based on the provided context.", []

    top = rrf[:top_n]
    context = build_context(chunks, top)

    prompt = f"""
Answer using ONLY the context.

Rules:
1. Explain in simple terms.
2. Do NOT repeat the question.
3. For process questions, explain inputs, main steps, and output.
4. Keep it concise (3-5 sentences).
5. If context is partial, give the best partial answer from context.

Question: {query}

Context:
{context}

Final answer:
""".strip()

    # FLAN-T5 base uses typical encoder length around 512 tokens.
    # Keep prompt within safe limits.
    inputs = _TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_DEVICE)

    try:
        with torch.no_grad():
            output_ids = _MODEL.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=4,
                do_sample=False,
                early_stopping=True,
                repetition_penalty=1.12,
                length_penalty=1.0,
            )
        raw = _TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        final = _clean_answer(raw)

        if _is_bad_answer(final, query):
            final = _extractive_fallback(query, chunks, top)

    except Exception:
        final = _extractive_fallback(query, chunks, top)

    return final, top
