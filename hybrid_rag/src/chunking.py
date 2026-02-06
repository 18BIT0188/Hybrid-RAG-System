from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text: str, min_tokens=200, max_tokens=400, overlap=50):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunk_ids = ids[start:end]
        if len(chunk_ids) >= min_tokens:
            chunk_text = tokenizer.decode(chunk_ids)
            chunks.append(chunk_text)
        if end == len(ids):
            break
        start = end - overlap
    return chunks
