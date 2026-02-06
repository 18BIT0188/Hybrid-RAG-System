import json, re, time, random, requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "hybrid-rag-assignment/1.0"}

def wc(text): return len(re.findall(r"\w+", text or ""))

def wiki_get(params, retries=5):
    for i in range(retries):
        try:
            r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** i) + random.random()); continue
            r.raise_for_status()
            if "application/json" not in (r.headers.get("Content-Type","").lower()):
                time.sleep((2 ** i) + random.random()); continue
            return r.json()
        except Exception:
            time.sleep((2 ** i) + random.random())
    return None

def random_titles(n=100):
    d = wiki_get({
        "action":"query","format":"json","list":"random",
        "rnnamespace":0,"rnlimit":n
    })
    return [x["title"] for x in d.get("query",{}).get("random",[])] if d else []

def get_page(title):
    d = wiki_get({
        "action":"query","format":"json","prop":"extracts|info",
        "explaintext":1,"inprop":"url","titles":title
    })
    if not d: return None
    pages = d.get("query",{}).get("pages",{})
    if not pages: return None
    p = next(iter(pages.values()))
    if "missing" in p: return None
    return {
        "pageid": p.get("pageid"),
        "title": p.get("title"),
        "url": p.get("fullurl"),
        "text": p.get("extract","")
    }

def main():
    with open("data/fixed_urls.json","r",encoding="utf-8") as f:
        fixed = json.load(f)

    fixed_urls = {x["url"] for x in fixed}
    out, seen_urls = [], set(fixed_urls)

    while len(out) < 300:
        for t in random_titles(100):
            p = get_page(t)
            if not p or not p["url"]: 
                continue
            if p["url"] in seen_urls:
                continue
            if wc(p["text"]) < 200:
                continue
            out.append({"title": p["title"], "url": p["url"]})
            seen_urls.add(p["url"])
            if len(out) >= 300:
                break

    with open("data/random_urls.json","w",encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(out)} random URLs -> data/random_urls.json")

if __name__ == "__main__":
    main()
