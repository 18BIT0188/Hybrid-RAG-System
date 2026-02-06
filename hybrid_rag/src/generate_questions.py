# src/generate_questions.py
import argparse
import json
import os
import random
import re
from collections import defaultdict

def load_corpus(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def split_sentences(text: str):
    # simple sentence splitter
    text = normalize_space(text)
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 30]

def title_to_question(title: str) -> str:
    return f"What is {title}?"

def pick_answer_sentence(sentences):
    # prefer medium-length informative sentence
    candidates = [s for s in sentences if 60 <= len(s) <= 240]
    if not candidates:
        candidates = sentences
    return random.choice(candidates) if candidates else None

def make_factual_q(title: str):
    return f"What is {title}?"

def make_comparative_q(t1: str, t2: str):
    return f"How does {t1} differ from {t2}?"

def make_inferential_q(title: str):
    return f"Why is {title} important?"

def make_multihop_q(t1: str, t2: str):
    return f"What connection can be made between {t1} and {t2}?"

def category_for_title(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["war", "empire", "revolution", "history"]):
        return "history"
    if any(k in t for k in ["economics", "inflation", "trade", "policy", "market"]):
        return "economics"
    if any(k in t for k in ["biology", "dna", "cancer", "virus", "immune"]):
        return "biology"
    if any(k in t for k in ["machine", "computer", "ai", "cryptography", "programming"]):
        return "technology"
    if any(k in t for k in ["india", "china", "states", "kingdom", "germany", "japan"]):
        return "geography"
    return "general"

def build_questions(corpus_rows, count=100, seed=42):
    random.seed(seed)

    # group chunks by URL (doc-level)
    docs = {}
    for r in corpus_rows:
        url = r.get("url")
        if not url:
            continue
        docs.setdefault(url, {
            "title": r.get("title", "Unknown"),
            "url": url,
            "chunks": []
        })
        txt = r.get("chunk_text") if "chunk_text" in r else r.get("text", "")
        if txt:
            docs[url]["chunks"].append(txt)

    doc_list = list(docs.values())
    if len(doc_list) < 20:
        raise ValueError(f"Need more documents in corpus. Found only {len(doc_list)}.")

    # build sentences
    for d in doc_list:
        joined = " ".join(d["chunks"][:3])
        d["sentences"] = split_sentences(joined)

    # category grouping
    by_cat = defaultdict(list)
    for d in doc_list:
        cat = category_for_title(d["title"])
        by_cat[cat].append(d)

    for c in by_cat:
        random.shuffle(by_cat[c])

    qas = []
    qid = 1

    target = {
        "factual": int(count * 0.45),
        "comparative": int(count * 0.20),
        "inferential": int(count * 0.20),
        "multi-hop": count - int(count * 0.45) - int(count * 0.20) - int(count * 0.20),
    }

    # ---------- factual ----------
    factual_docs = doc_list.copy()
    random.shuffle(factual_docs)

    for d in factual_docs:
        if target["factual"] <= 0:
            break

        ans = pick_answer_sentence(d["sentences"])
        if not ans:
            continue

        qas.append({
            "id": f"q{qid:03d}",
            "question": make_factual_q(d["title"]),
            "ground_truth": ans,
            "source_urls": [d["url"]],
            "source_titles": [d["title"]],
            "category": "factual"
        })
        qid += 1
        target["factual"] -= 1

    # ---------- comparative ----------
    cat_keys = [c for c in by_cat if len(by_cat[c]) >= 2]

    while target["comparative"] > 0 and cat_keys:
        c = random.choice(cat_keys)
        d1, d2 = random.sample(by_cat[c], 2)

        ans1 = pick_answer_sentence(d1["sentences"]) or d1["title"]
        ans2 = pick_answer_sentence(d2["sentences"]) or d2["title"]

        qas.append({
            "id": f"q{qid:03d}",
            "question": make_comparative_q(d1["title"], d2["title"]),
            "ground_truth": f"{d1['title']}: {ans1} || {d2['title']}: {ans2}",
            "source_urls": [d1["url"], d2["url"]],
            "source_titles": [d1["title"], d2["title"]],
            "category": "comparative"
        })
        qid += 1
        target["comparative"] -= 1

    # ---------- inferential ----------
    infer_docs = doc_list.copy()
    random.shuffle(infer_docs)

    for d in infer_docs:
        if target["inferential"] <= 0:
            break

        ans = pick_answer_sentence(d["sentences"])
        if not ans:
            continue

        qas.append({
            "id": f"q{qid:03d}",
            "question": make_inferential_q(d["title"]),
            "ground_truth": ans,
            "source_urls": [d["url"]],
            "source_titles": [d["title"]],
            "category": "inferential"
        })
        qid += 1
        target["inferential"] -= 1

    # ---------- multi-hop ----------
    pool = doc_list.copy()
    random.shuffle(pool)
    i = 0

    while target["multi-hop"] > 0 and i + 1 < len(pool):
        d1, d2 = pool[i], pool[i + 1]
        i += 2

        ans1 = pick_answer_sentence(d1["sentences"]) or d1["title"]
        ans2 = pick_answer_sentence(d2["sentences"]) or d2["title"]

        qas.append({
            "id": f"q{qid:03d}",
            "question": make_multihop_q(d1["title"], d2["title"]),
            "ground_truth": f"{d1['title']}: {ans1} || {d2['title']}: {ans2}",
            "source_urls": [d1["url"], d2["url"]],
            "source_titles": [d1["title"], d2["title"]],
            "category": "multi-hop"
        })
        qid += 1
        target["multi-hop"] -= 1

    # ---------- GUARANTEED fallback ----------
    random.shuffle(doc_list)

    while len(qas) < count:
        d = random.choice(doc_list)
        ans = pick_answer_sentence(d["sentences"])
        if not ans:
            continue

        qas.append({
            "id": f"q{qid:03d}",
            "question": make_factual_q(d["title"]),
            "ground_truth": ans,
            "source_urls": [d["url"]],
            "source_titles": [d["title"]],
            "category": "factual"
        })
        qid += 1

    return qas[:count]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="eval/questions_100.json")
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("Loading corpus...")
    rows = load_corpus(args.corpus)
    print("Corpus rows:", len(rows))

    print("Generating questions...")
    qas = build_questions(rows, count=args.count)
    print("Generated:", len(qas))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(qas, f, indent=2, ensure_ascii=False)

    print("Saved to:", args.out)


if __name__ == "__main__":
    main()

