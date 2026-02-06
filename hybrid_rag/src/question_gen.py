# src/question_gen.py
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on at for from by with as
this that these those it its their his her they them we you i
""".split())

def _sentences(text: str) -> List[str]:
    # simple sentence split
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) > 20]

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _keywords(text: str, max_k: int = 12) -> List[str]:
    toks = _normalize(text).split()
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 3]
    freq = defaultdict(int)
    for t in toks:
        freq[t] += 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_k]]

def load_chunks(corpus_path="data/corpus.jsonl") -> List[Dict]:
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    chunks.append(obj)
            except json.JSONDecodeError:
                # skip malformed line
                continue
    return chunks


def build_page_texts(chunks: List[Dict]) -> Dict[str, Dict]:
    """
    Merge chunk_text by page identifier for question generation.
    Supports multiple schema variants:
      - pageid
      - page_id
      - source_id
      - url (fallback key)
    """
    pages: Dict[str, Dict] = {}

    for c in chunks:
        title = c.get("title") or c.get("page_title") or "Unknown"
        url = c.get("url") or c.get("source_url") or ""

        # robust page id fallback chain
        pid_raw = (
            c.get("pageid")
            or c.get("page_id")
            or c.get("source_id")
            or c.get("doc_id")
            or url
            or title
        )
        pid = str(pid_raw)

        text = c.get("chunk_text") or c.get("text") or c.get("content") or ""
        if not text:
            continue

        if pid not in pages:
            pages[pid] = {
                "pageid": pid,
                "title": title,
                "url": url,
                "texts": [],
            }

        pages[pid]["texts"].append(text)

        # keep best title/url if missing
        if pages[pid].get("title") in ("", "Unknown") and title:
            pages[pid]["title"] = title
        if not pages[pid].get("url") and url:
            pages[pid]["url"] = url

    # concatenate page text
    for pid in pages:
        pages[pid]["full_text"] = " ".join(pages[pid]["texts"])

    return pages


def gen_factual(p: Dict) -> Tuple[str, str]:
    # Use first informative sentence as "ground truth"
    sents = _sentences(p["full_text"])
    if not sents:
        return "", ""
    gt = sents[0]
    q = f"What is {p['title']}?"
    return q, gt

def gen_inferential(p: Dict) -> Tuple[str, str]:
    # Find because/therefore sentence
    sents = _sentences(p["full_text"])
    candidates = [s for s in sents if re.search(r"\b(because|therefore|thus|as a result)\b", s.lower())]
    if not candidates:
        return "", ""
    gt = random.choice(candidates)
    q = f"Why does {p['title']} matter or happen? Explain briefly."
    return q, gt

def gen_comparative(a: Dict, b: Dict) -> Tuple[str, str]:
    # Ground truth = 1-2 sentences from each page
    sa = _sentences(a["full_text"])
    sb = _sentences(b["full_text"])
    if not sa or not sb:
        return "", ""
    gt = f"{a['title']}: {sa[0]}  {b['title']}: {sb[0]}"
    q = f"How does {a['title']} differ from {b['title']}?"
    return q, gt

def gen_multihop(a: Dict, b: Dict) -> Tuple[str, str]:
    # Create question that needs both pages: shared keyword bridge
    ka = set(_keywords(a["full_text"], 20))
    kb = set(_keywords(b["full_text"], 20))
    common = list(ka.intersection(kb))
    if not common:
        return "", ""
    bridge = random.choice(common)
    # ground truth = sentence from each page containing bridge keyword if possible
    sa = [s for s in _sentences(a["full_text"]) if bridge in _normalize(s)]
    sb = [s for s in _sentences(b["full_text"]) if bridge in _normalize(s)]
    if not sa or not sb:
        return "", ""
    gt = f"From {a['title']}: {sa[0]}  From {b['title']}: {sb[0]}"
    q = f"Using both topics, explain how '{bridge}' relates to {a['title']} and {b['title']}."
    return q, gt

def gen_unanswerable(p: Dict) -> Tuple[str, str]:
    # Should not exist in Wikipedia context => force I don't know
    q = f"What is the phone number of {p['title']}?"
    gt = "I don't know"
    return q, gt

def generate_questions(
    n_total: int = 100,
    seed: int = 42,
    include_unanswerable: int = 10,
    corpus_path="data/corpus.jsonl",
    out_path="eval/questions_100.json",
):
    random.seed(seed)
    Path("eval").mkdir(exist_ok=True)

    chunks = load_chunks(corpus_path)
    pages = list(build_page_texts(chunks).values())
    random.shuffle(pages)

    # quotas
    n_fact = 40
    n_inf = 20
    n_comp = 15
    n_multi = 15
    n_unans = include_unanswerable
    # adjust if n_total differs
    n_sum = n_fact + n_inf + n_comp + n_multi + n_unans
    if n_sum != n_total:
        # simple adjust factual
        n_fact += (n_total - n_sum)

    qs = []
    used = set()

    def add_item(q, gt, pids, urls, cat):
        if not q or not gt:
            return
        key = (q.strip().lower(), tuple(sorted(urls)))
        if key in used:
            return
        used.add(key)
        qs.append({
            "id": f"q{len(qs)+1:03d}",
            "question": q,
            "ground_truth": gt,
            "source_pageids": pids,
            "source_urls": urls,
            "category": cat,
        })

    # factual
    i = 0
    while len([x for x in qs if x["category"] == "factual"]) < n_fact and i < len(pages):
        p = pages[i]
        q, gt = gen_factual(p)
        add_item(q, gt, [p["pageid"]], [p["url"]], "factual")
        i += 1

    # inferential
    tries = 0
    while len([x for x in qs if x["category"] == "inferential"]) < n_inf and tries < 500:
        p = random.choice(pages)
        q, gt = gen_inferential(p)
        add_item(q, gt, [p["pageid"]], [p["url"]], "inferential")
        tries += 1

    # comparative (two pages)
    tries = 0
    while len([x for x in qs if x["category"] == "comparative"]) < n_comp and tries < 800:
        a, b = random.sample(pages, 2)
        q, gt = gen_comparative(a, b)
        add_item(q, gt, [a["pageid"], b["pageid"]], [a["url"], b["url"]], "comparative")
        tries += 1

    # multi-hop
    tries = 0
    while len([x for x in qs if x["category"] == "multi_hop"]) < n_multi and tries < 1200:
        a, b = random.sample(pages, 2)
        q, gt = gen_multihop(a, b)
        add_item(q, gt, [a["pageid"], b["pageid"]], [a["url"], b["url"]], "multi_hop")
        tries += 1

    # unanswerable
    if n_unans > 0:
        for p in random.sample(pages, min(n_unans, len(pages))):
            q, gt = gen_unanswerable(p)
            add_item(q, gt, [p["pageid"]], [p["url"]], "unanswerable")

    # trim / ensure exact count
    qs = qs[:n_total]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(qs)} questions to {out_path}")

if __name__ == "__main__":
    generate_questions()
