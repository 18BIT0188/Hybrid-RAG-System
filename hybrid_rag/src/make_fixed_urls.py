# src/make_fixed_urls.py
# Generate a fixed set of 200 unique Wikipedia URLs (>=200 words each), diverse by category.
# Output: data/fixed_urls.json

import json
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "hybrid-rag-assignment/1.0 (student project; contact: your-email@example.com)"
}

# Category seeds for diversity
SEED_CATEGORIES: Dict[str, List[str]] = {
    "Science": ["Physics", "Chemistry", "Biology", "Astronomy", "Geology"],
    "Technology": ["Artificial intelligence", "Computer science", "Software engineering", "Cryptography"],
    "History": ["Ancient history", "Medieval history", "World War II", "Renaissance"],
    "Geography": ["Countries", "Rivers", "Mountains", "Deserts"],
    "Politics": ["Democracy", "Constitution", "United Nations", "Elections"],
    "Economics": ["Inflation", "Macroeconomics", "International trade", "Monetary policy"],
    "Mathematics": ["Algebra", "Calculus", "Statistics", "Graph theory"],
    "Medicine": ["Immunology", "Epidemiology", "Cancer", "Public health"],
    "Culture": ["Literature", "Music", "Painting", "Cinema"],
    "Sports": ["Olympic Games", "Association football", "Cricket", "Tennis"],
    "Environment": ["Climate change", "Biodiversity", "Deforestation", "Renewable energy"],
    "Philosophy": ["Ethics", "Metaphysics", "Logic", "Epistemology"],
}


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def wiki_get(params: Dict, retries: int = 6, base_sleep: float = 0.6) -> Optional[Dict]:
    """
    Robust GET to Wikipedia API with retry/backoff.
    Handles rate limits / transient failures / non-JSON responses.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)

            # Retry transient/rate-limit errors
            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" not in content_type:
                # Sometimes HTML/empty responses show up briefly; retry
                sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
                time.sleep(sleep_s)
                continue

            return resp.json()

        except Exception:
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
            time.sleep(sleep_s)

    return None


def get_page(title: str) -> Optional[Dict]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "titles": title,
    }
    data = wiki_get(params)
    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    if "missing" in page:
        return None

    return {
        "pageid": page.get("pageid"),
        "title": page.get("title"),
        "url": page.get("fullurl"),
        "extract": page.get("extract", ""),
    }


def random_titles(n: int = 100) -> List[str]:
    # Wikipedia max rnlimit for normal users is 500
    n = min(max(1, n), 500)
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": n,
    }
    data = wiki_get(params)
    if not data:
        return []
    return [x["title"] for x in data.get("query", {}).get("random", []) if "title" in x]


def linked_titles(title: str, limit: int = 200) -> List[str]:
    # Fetch outgoing links from a seed article to stay topically coherent
    limit = min(max(1, limit), 500)
    params = {
        "action": "query",
        "format": "json",
        "prop": "links",
        "pllimit": limit,
        "titles": title,
    }
    data = wiki_get(params)
    if not data:
        return []

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return []

    page = next(iter(pages.values()))
    links = page.get("links", [])
    return [l["title"] for l in links if "title" in l]


def add_if_valid(
    title: str,
    chosen: List[Dict[str, str]],
    seen_pageids: Set[int],
    min_words: int
) -> bool:
    page = get_page(title)
    if not page:
        return False

    pageid = page.get("pageid")
    url = page.get("url")
    if not pageid or not url:
        return False
    if pageid in seen_pageids:
        return False
    if word_count(page.get("extract", "")) < min_words:
        return False

    chosen.append({"title": page["title"], "url": url})
    seen_pageids.add(pageid)
    return True


def build_fixed_urls(
    target: int = 200,
    min_words: int = 200,
    out_file: str = "data/fixed_urls.json",
) -> None:
    chosen: List[Dict[str, str]] = []
    seen_pageids: Set[int] = set()

    # Balanced per-category quota for diversity
    n_categories = len(SEED_CATEGORIES)
    cat_quota = max(10, target // n_categories)  # around 16 for 200/12
    per_cat_count = defaultdict(int)

    print(f"Starting generation: target={target}, min_words={min_words}, per_category_quota={cat_quota}")

    # Phase 1: Category-guided collection (diversity)
    for category, seeds in SEED_CATEGORIES.items():
        random.shuffle(seeds)
        print(f"\n[Category] {category}")

        for seed in seeds:
            candidates = [seed] + linked_titles(seed, limit=250)
            random.shuffle(candidates)

            for cand in candidates:
                if per_cat_count[category] >= cat_quota:
                    break
                added = add_if_valid(
                    title=cand,
                    chosen=chosen,
                    seen_pageids=seen_pageids,
                    min_words=min_words
                )
                if added:
                    per_cat_count[category] += 1
                    if len(chosen) % 10 == 0:
                        print(f"Collected {len(chosen)}/{target}")

                    if len(chosen) >= target:
                        break

            if len(chosen) >= target:
                break

            # gentle pacing
            time.sleep(0.15)

        if len(chosen) >= target:
            break

    # Phase 2: Fill remaining with random pages
    print("\nFilling remaining with random pages...")
    while len(chosen) < target:
        titles = random_titles(120)
        if not titles:
            time.sleep(1.0)
            continue

        for t in titles:
            added = add_if_valid(
                title=t,
                chosen=chosen,
                seen_pageids=seen_pageids,
                min_words=min_words
            )
            if added and len(chosen) % 10 == 0:
                print(f"Collected {len(chosen)}/{target}")

            if len(chosen) >= target:
                break

        time.sleep(0.2)

    # Exact size
    chosen = chosen[:target]

    # Save
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(chosen)} URLs to {out_file}")


if __name__ == "__main__":
    # As per assignment, fixed set should remain constant across indexing runs.
    # Run once, commit data/fixed_urls.json, and reuse it.
    build_fixed_urls(target=200, min_words=200, out_file="data/fixed_urls.json")
