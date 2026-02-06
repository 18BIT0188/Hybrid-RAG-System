import json
from pathlib import Path
from typing import Dict, Any

from .chunking import chunk_text
from .wiki_collect import fetch_page_plaintext


DATA_DIR = Path("data")
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
FIXED_URLS_PATH = DATA_DIR / "fixed_urls.json"
RANDOM_URLS_PATH = DATA_DIR / "random_urls.json"


def _load_urls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Empty or partially written file → treat as no URLs
            return []
    # Normalise: allow records with at least title+url
    norm: list[dict] = []
    for item in data:
        if "title" in item and "url" in item:
            norm.append(item)
    return norm


def _fetch_plaintext_for_url(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Given an entry with at least a `title` or `url`, fetch Wikipedia plain text.
    Prefers using the title for robustness.
    """
    title = entry.get("title")
    if title:
        page = fetch_page_plaintext(title)
        if page:
            return page

    # Fallback: derive title from URL path if possible
    url = entry.get("url", "")
    if "wikipedia.org/wiki/" in url:
        derived_title = url.rsplit("/", 1)[-1].replace("_", " ")
        page = fetch_page_plaintext(derived_title)
        if page:
            return page

    return None


def build_corpus(min_tokens: int = 200, max_tokens: int = 400, overlap: int = 50) -> None:
    """
    Build `data/corpus.jsonl` from `fixed_urls.json` + `random_urls.json`.

    Each line in `corpus.jsonl` is a JSON object:
    {
        "title": ...,
        "url": ...,
        "chunk_text": ...
    }
    """
    fixed_urls = _load_urls(FIXED_URLS_PATH)
    random_urls = _load_urls(RANDOM_URLS_PATH)
    all_entries = fixed_urls + random_urls

    if not all_entries:
        raise RuntimeError(
            "No URLs found in `data/fixed_urls.json` or `data/random_urls.json`. "
            "Populate at least one of these files with Wikipedia titles/URLs first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as out_f:
        for entry in all_entries:
            page = _fetch_plaintext_for_url(entry)
            if not page:
                continue
            text = page.get("text", "")
            if not text:
                continue

            chunks = chunk_text(
                text,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap=overlap,
            )
            for chunk in chunks:
                rec = {
                    "title": page["title"],
                    "url": page["url"],
                    "chunk_text": chunk,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_corpus()

