import hashlib, time
from pathlib import Path
from typing import List, Dict
import httpx
from selectolax.parser import HTMLParser
from trafilatura import extract
from openai import OpenAI

client = OpenAI()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def ensure_db():
    (DATA_DIR / "chroma").mkdir(parents=True, exist_ok=True)

def _fetch_clean(url: str):
    try:
        r = httpx.get(url, timeout=45, follow_redirects=True)
        r.raise_for_status()
        title_node = HTMLParser(r.text).css_first("title")
        title = title_node.text() if title_node else url
        text = extract(r.text, include_comments=False, include_tables=False, url=url) or ""
        if not text:
            html = HTMLParser(r.text)
            for s in html.css("script,style,nav,footer,header"):
                s.decompose()
            text = html.text(strip=True)
        return title.strip(), text.strip()
    except Exception:
        return None, ""

def _chunk(text: str, max_tokens=1000, overlap=120):
    words = text.split()
    if not words:
        return
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i:i+max_tokens])

def _existing_urls(collection) -> set:
    res = collection.get(include=["metadatas", "ids"])
    metas = res.get("metadatas") or []
    urls = set()
    for m in metas:
        if m and "url" in m:
            urls.add(m["url"])
    return urls

def _delete_url(collection, url: str):
    res = collection.get(include=["metadatas", "ids"])
    ids_to_delete = []
    for rid, m in zip(res.get("ids", []), res.get("metadatas", [])):
        if m and m.get("url") == url:
            ids_to_delete.append(rid)
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

def _upsert_url(collection, url: str) -> int:
    title, text = _fetch_clean(url)
    if not text:
        return 0
    docs, ids, metas = [], [], []
    for n, chunk in enumerate(_chunk(text)):
        doc_id = hashlib.sha1(f"{url}::{n}".encode()).hexdigest()
        docs.append(chunk)
        ids.append(doc_id)
        metas.append({"url": url, "title": title, "chunk": n})
    # Embeddings
    emb = client.embeddings.create(model="text-embedding-3-large", input=docs)
    vecs = [e.embedding for e in emb.data]
    # Clean existing chunks for this URL, then add
    try:
        _delete_url(collection, url)
    except Exception:
        pass
    collection.add(documents=docs, embeddings=vecs, metadatas=metas, ids=ids)
    return len(docs)

def reindex(incoming_urls: List[str], collection) -> Dict:
    existing = _existing_urls(collection)
    incoming = set(incoming_urls)

    removed = existing - incoming
    added = incoming - existing
    stayed = incoming & existing

    # Remove obsolete URLs
    for url in removed:
        _delete_url(collection, url)

    # (Re)upsert everything in incoming; small scale = simplest + fresh
    total_chunks = 0
    for url in incoming:
        total_chunks += _upsert_url(collection, url)
        time.sleep(0.4)  # be polite

    return {
        "added_urls_count": len(added),
        "removed_urls_count": len(removed),
        "total_indexed_chunks": total_chunks,
    }
