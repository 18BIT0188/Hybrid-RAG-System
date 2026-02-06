import time
import random
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"

def fetch_page_plaintext(title: str, max_retries: int = 5) -> dict | None:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "titles": title,
        "redirects": 1,
        "origin": "*",
    }

    headers = {
        "User-Agent": "hybrid-rag-assignment/1.0 (educational project)",
        "Accept": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(WIKI_API, params=params, headers=headers, timeout=30)

            # Retry for transient server/rate-limit errors
            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(2 ** attempt, 20) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()

            ctype = resp.headers.get("Content-Type", "").lower()
            if "application/json" not in ctype:
                # sometimes returns HTML/error page
                sleep_s = min(2 ** attempt, 20) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue

            data = resp.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page = next(iter(pages.values()))
            if "missing" in page:
                return None

            text = (page.get("extract") or "").strip()
            if not text:
                return None

            return {
                "title": page.get("title", title),
                "url": page.get("fullurl"),
                "pageid": page.get("pageid"),
                "text": text,
            }

        except (requests.RequestException, ValueError):
            # ValueError covers json decode errors
            if attempt == max_retries:
                return None
            sleep_s = min(2 ** attempt, 20) + random.uniform(0, 0.5)
            time.sleep(sleep_s)

    return None
