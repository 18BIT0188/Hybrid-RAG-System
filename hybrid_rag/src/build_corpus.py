import json, re, requests
from transformers import AutoTokenizer

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "hybrid-rag-assignment/1.0"}
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def fetch_text(title):
    r = requests.get(WIKI_API, params={
        "action":"query","format":"json","prop":"extracts","explaintext":1,"titles":title
    }, headers=HEADERS, timeout=30)
    d = r.json()
    p = next(iter(d["query"]["pages"].values()))
    txt = p.get("extract","")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def chunk_text(text, min_t=200, max_t=400, overlap=50):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + max_t, len(ids))
        part = ids[start:end]
        if len(part) >= min_t:
            chunks.append(tokenizer.decode(part))
        if end == len(ids): break
        start = end - overlap
    return chunks

def main():
    with open("data/fixed_urls.json","r",encoding="utf-8") as f:
        fixed = json.load(f)
    with open("data/random_urls.json","r",encoding="utf-8") as f:
        rand = json.load(f)

    urls = fixed + rand
    assert len(urls) == 500, f"Expected 500 URLs, got {len(urls)}"

    records = []
    for doc_i, item in enumerate(urls):
        title, url = item["title"], item["url"]
        text = fetch_text(title)
        chunks = chunk_text(text, 200, 400, 50)
        for j, ch in enumerate(chunks):
            records.append({
                "chunk_id": f"doc{doc_i:04d}_chunk{j:03d}",
                "title": title,
                "url": url,
                "text": ch
            })

    with open("data/corpus.jsonl","w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Docs: {len(urls)} | Chunks: {len(records)} -> data/corpus.jsonl")

if __name__ == "__main__":
    main()
