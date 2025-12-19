from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

MAX_INGEST_URLS = 200


def fetch_sitemap_urls(base_url: str) -> list[str]:
    sitemap_url = urljoin(base_url, "/sitemap.xml")

    try:
        res = requests.get(sitemap_url, timeout=10)
        res.raise_for_status()
    except Exception:
        return []

    urls = []
    try:
        root = ET.fromstring(res.text)
        for loc in root.iter():
            if loc.tag.endswith("loc") and loc.text:
                urls.append(loc.text.strip())
    except Exception:
        return []

    return urls


def resolve_urls_by_scope(base_url: str, scope: str) -> list[str]:
    """
    scope に応じて ingest 対象の URL 一覧を返す
    - sitemap.xml 優先
    - URL 数は MAX_INGEST_URLS まで
    """
    if scope == "single":
        return [base_url]

    # ① sitemap を試す
    sitemap_urls = fetch_sitemap_urls(base_url)
    if sitemap_urls:
        base_path = urlparse(base_url).path.rstrip("/")
        filtered = []

        for url in sitemap_urls:
            path = urlparse(url).path.rstrip("/")

            if scope == "all":
                if path.startswith(base_path):
                    filtered.append(url)

            elif scope == "one-level":
                if path.count("/") <= base_path.count("/") + 1:
                    filtered.append(url)

            if len(filtered) >= MAX_INGEST_URLS:
                break

        return filtered or [base_url]

    # ② sitemap が無ければ HTML クロール
    try:
        res = requests.get(base_url, timeout=10)
        res.raise_for_status()
    except Exception:
        return [base_url]

    soup = BeautifulSoup(res.text, "html.parser")
    base_netloc = urlparse(base_url).netloc
    base_path = urlparse(base_url).path.rstrip("/")

    urls = []

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)

        if parsed.netloc != base_netloc:
            continue

        path = parsed.path.rstrip("/")

        if scope == "one-level":
            if path.count("/") <= base_path.count("/") + 1:
                urls.append(href)

        elif scope == "all":
            if path.startswith(base_path):
                urls.append(href)

        if len(urls) >= MAX_INGEST_URLS:
            break

    return urls or [base_url]
