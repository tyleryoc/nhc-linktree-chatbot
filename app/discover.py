import httpx
from selectolax.parser import HTMLParser

def discover_links(linktree_url: str):
    r = httpx.get(linktree_url, timeout=30)
    r.raise_for_status()
    html = HTMLParser(r.text)
    links = set()
    # Collect visible outbound links
    for a in html.css("a"):
        href = a.attributes.get("href", "")
        if href and href.startswith("http"):
            links.add(href)
    # Remove obvious non-destination artifacts if needed
    # (You can customize pruning here.)
    return sorted(links)
