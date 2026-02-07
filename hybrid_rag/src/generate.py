# src/generate.py
from typing import Dict, List, Tuple

# Optional torch/transformers imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    TORCH_AVAILABLE = False

MODEL_NAME = "google/flan-t5-base"

_TOKENIZER = None
_MODEL = None
_DEVICE = "cpu"


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def build_context(chunks: List[dict], selected: List[Tuple[int, float]], max_chars_per_chunk: int = 700) -> str:
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


def _load_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if not TORCH_AVAILABLE:
        return False
    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return True


def _extractive_fallback(query: str, chunks: List[dict], top: List[Tuple[int, float]]) -> str:
    """
    No-LLM fallback: return concise grounded answer from top snippet(s).
    """
    if not top:
        return "I don't know."

    first_idx = top[0][0]
    if first_idx < 0 or first_idx >= len(chunks):
        return "I don't know."

    txt = _normalize_whitespace(chunks[first_idx].get("chunk_text") or chunks[first_idx].get("text") or "")
    if not txt:
        return "I don't know."

    # pick first 2-3 sentences
    sents = [s.strip() for s in txt.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    if not sents:
        return txt[:350]

    ans = ". ".join(sents[:3]).strip()
    if not ans.endswith("."):
        ans += "."
    return ans[:500]


def _clean_answer(raw: str) -> str:
    txt = _normalize_whitespace(raw)
    if not txt:
        return "I don't know."
    if len(txt) > 700:
        txt = txt[:700].rsplit(" ", 1)[0].strip()
    return txt


def answer(query: str, retrieved: Dict, top_n: int = 5):
    """
    Returns:
      (final_answer: str, selected_chunks: List[(chunk_idx, score)])
    """
    chunks = retrieved.get("chunks", [])
    top = retrieved.get("rrf", [])[:top_n]

    # If model deps missing, fallback
    if not _load_model():
        return _extractive_fallback(query, chunks, top), top

    context = build_context(chunks, top)

    prompt = f"""
Answer using ONLY the context.
Rules:
1. Explain simply.
2. 3-5 sentences.
3. If context is partial, answer only what is supported.

Question: {query}

Context:
{context}

Final answer:
""".strip()

    try:
        inputs = _TOKENIZER(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(_DEVICE)

        with torch.no_grad():
            output_ids = _MODEL.generate(
                **inputs,
                max_new_tokens=120,
                num_beams=3,
                do_sample=False,
                repetition_penalty=1.1,
            )

        raw = _TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
        final = _clean_answer(raw)
        if final.lower() in {"i don't know", "i do not know"}:
            # grounded fallback instead of empty refusal
            final = _extractive_fallback(query, chunks, top)
        return final, top

    except Exception:
        # robust fallback if generation fails at runtime
        return _extractive_fallback(query, chunks, top), top
