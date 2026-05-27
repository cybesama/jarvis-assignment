"""
Async crawler for jarvislabs.ai.
Focused crawl: docs, faq, pricing, gpu-guides, templates, kb.
Skips blog, marketing pages, and other noise to keep the KB tight.
"""

import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from typing import Set, List, Dict
import hashlib

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from config import settings


# Only crawl these path prefixes — keeps KB focused on support content
_ALLOW_PATH_PREFIXES = (
    "/docs",
    "/faq",
    "/pricing",
    "/templates",
    "/gpu-cloud",
    "/gpu-guides",
    "/support",
    "/kb",
    "/",          # homepage
)

_SKIP_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".ico",
    ".woff", ".woff2", ".ttf", ".css", ".js",
}

_SKIP_PATTERNS = re.compile(
    r"/(cdn-cgi|_next/static|static/|favicon|robots\.txt|sitemap|login|signup|dashboard)"
)

# Hard cap — keeps ChromaDB and embedding time manageable
MAX_PAGES = 300


def _clean_url(url: str) -> str:
    url, _ = urldefrag(url)
    return url.rstrip("/")


def _is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if settings.SCRAPE_DOMAIN_ALLOW not in parsed.netloc:
        return False
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return False
    if _SKIP_PATTERNS.search(path):
        return False
    # Only follow allowed path prefixes
    if not any(path.startswith(p) for p in _ALLOW_PATH_PREFIXES):
        return False
    return True


def _extract_text(html: str, url: str) -> Dict | None:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["nav", "footer", "script", "style", "noscript",
                     "header", "aside", "form", "button", "iframe"]):
        tag.decompose()

    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""

    meta_el = soup.find("meta", attrs={"name": "description"})
    meta = meta_el.get("content", "") if meta_el else ""

    content_el = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|docs", re.I))
        or soup.find("body")
    )
    if not content_el:
        return None

    raw = content_el.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", raw).strip()

    if len(text) < 150:   # skip near-empty pages
        return None

    return {
        "id": hashlib.md5(url.encode()).hexdigest(),
        "url": url,
        "title": title,
        "meta": meta,
        "text": text,
    }


def _extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        full = _clean_url(urljoin(base_url, a["href"].strip()))
        links.append(full)
    return links


async def crawl(output_dir: Path = None) -> List[Dict]:
    output_dir = output_dir or settings.RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    visited: Set[str] = set()
    frontier: List[tuple] = [(_clean_url(s), 0) for s in settings.SCRAPE_SEEDS]
    results: List[Dict] = []
    sem = asyncio.Semaphore(settings.SCRAPE_CONCURRENCY)

    async def fetch(client: httpx.AsyncClient, url: str, depth: int):
        if url in visited or len(results) >= MAX_PAGES:
            return
        visited.add(url)

        async with sem:
            try:
                r = await client.get(url, follow_redirects=True, timeout=15)
            except Exception as exc:
                logger.warning(f"skip {url}: {exc}")
                return

        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            return

        doc = _extract_text(r.text, url)
        if doc:
            results.append(doc)
            logger.info(f"[d={depth}] ({len(results)}/{MAX_PAGES}) {url}")

        if depth < settings.SCRAPE_MAX_DEPTH:
            for link in _extract_links(r.text, url):
                if link not in visited and _is_allowed(link):
                    frontier.append((link, depth + 1))

    headers = {"User-Agent": "Jarvina-KnowledgeBot/1.0"}
    async with httpx.AsyncClient(headers=headers) as client:
        while frontier and len(results) < MAX_PAGES:
            batch, frontier = frontier[:settings.SCRAPE_CONCURRENCY * 2], frontier[settings.SCRAPE_CONCURRENCY * 2:]
            await asyncio.gather(*[fetch(client, url, d) for url, d in batch])

    out = output_dir / "jarvislabs_raw.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"Crawl done: {len(results)} pages saved to {out}")
    return results


if __name__ == "__main__":
    asyncio.run(crawl())
