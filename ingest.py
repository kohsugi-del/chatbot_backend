# ingest.py
import os, re, time, hashlib, argparse
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
load_dotenv()

# OpenAI（新SDK）
from openai import OpenAI

# Supabase
from supabase import create_client, Client


# ==============
# 設定
# ==============
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))

UA = os.getenv(
    "INGEST_UA",
    "Mozilla/5.0 (compatible; QwestIngestBot/1.0; +https://qwest.co.jp)"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
)

session = requests.Session()
session.headers.update({"User-Agent": UA})

# ===== 共通フィルタ設定（必要なら env で調整してもOK） =====
DENY_PATTERNS = [
    r"/page/\d+/?$",   # ページネーション
    r"/tag/",
    r"/category/",
    r"/wp-content/",
    r"/wp-json/",
]
DENY_QUERY = True      # ?つきURLは除外
DENY_FRAGMENT = True   # #付きURLは除外


# ==============
# ユーティリティ
# ==============
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def normalize_url(u: str) -> str:
    u = u.split("#")[0]
    if u.endswith("/"):
        return u
    parsed = urlparse(u)
    if parsed.path == "":
        return u + "/"
    return u

def is_allowed(url: str, base_host: str, allowed_paths: list[str]) -> bool:
    p = urlparse(url)

    # ドメインチェック
    if p.netloc != base_host:
        return False

    # クエリ除外
    if DENY_QUERY and p.query:
        return False

    # フラグメント除外（normalize_urlで落としているが念のため）
    if DENY_FRAGMENT and p.fragment:
        return False

    path = p.path or "/"

    # deny パターン
    for pat in DENY_PATTERNS:
        if re.search(pat, path):
            return False

    # allow 判定
    if not allowed_paths:
        return True
    return any(path.startswith(ap) for ap in allowed_paths)

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        absu = urljoin(base_url, href)
        absu = normalize_url(absu)
        links.append(absu)
    return links

def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "")

    main = soup.find("main")
    node = main if main else soup.body
    text = node.get_text("\n", strip=True) if node else soup.get_text("\n", strip=True)

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text

def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in res.data]


# ==============
# Supabase: state
# ==============
def state_get(site_id: int) -> dict:
    r = supabase.table("ingest_state").select("*").eq("site_id", site_id).execute()
    if r.data:
        return r.data[0]
    init = {
        "site_id": site_id,
        "cursor": 0,
        "total": 0,
        "status": "idle",
        "last_url": None,
        "last_error": None,
    }
    supabase.table("ingest_state").insert(init).execute()
    return init

def state_update(site_id: int, **kwargs):
    kwargs["updated_at"] = "now()"
    supabase.table("ingest_state").update(kwargs).eq("site_id", site_id).execute()


# ==============
# Supabase: documents upsert
# ==============
def upsert_documents(rows: list[dict]):
    supabase.table("documents").upsert(rows, on_conflict="site_id,url,chunk_index").execute()


# ==============
# Crawl: sitemap優先 → 無ければ簡易BFS
# ==============
def fetch_sitemap_urls(seed_url: str, allowed_paths: list[str], max_pages: int) -> list[str]:
    parsed = urlparse(seed_url)
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    try:
        r = session.get(sitemap_url, timeout=TIMEOUT)
        if r.status_code != 200 or "<urlset" not in r.text:
            return []
        soup = BeautifulSoup(r.text, "xml")
        locs = [loc.get_text(strip=True) for loc in soup.select("url > loc")]
        base_host = parsed.netloc
        urls = []
        for u in locs:
            u = normalize_url(u)
            if is_allowed(u, base_host, allowed_paths):
                urls.append(u)
            if len(urls) >= max_pages:
                break
        return urls
    except Exception:
        return []

def bfs_crawl(seed_url: str, allowed_paths: list[str], max_pages: int) -> list[str]:
    parsed = urlparse(seed_url)
    base_host = parsed.netloc

    q = [normalize_url(seed_url)]
    seen = set()
    out = []

    while q and len(out) < max_pages:
        u = q.pop(0)
        if u in seen:
            continue
        seen.add(u)

        if not is_allowed(u, base_host, allowed_paths):
            continue

        out.append(u)

        try:
            r = session.get(u, timeout=TIMEOUT)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
            for link in extract_links(r.text, u):
                if link not in seen and is_allowed(link, base_host, allowed_paths):
                    q.append(link)
        except Exception:
            continue

    return out


# ==============
# メイン ingest（単体）
# ==============
def run_ingest(
    site_id: int,
    seed_url: str,
    max_pages: int,
    allowed_paths: list[str],
    batch_size: int,
    sleep_sec: float,
    max_chars: int,
    overlap: int,
    resume_from: int | None,
    dry_run: bool,
):
    st = state_get(site_id)

    if resume_from is not None:
        st["cursor"] = resume_from

    state_update(site_id, status="running", last_error=None)

    urls = fetch_sitemap_urls(seed_url, allowed_paths, max_pages)
    if not urls:
        urls = bfs_crawl(seed_url, allowed_paths, max_pages)

    total = len(urls)
    state_update(site_id, total=total)

    if total == 0:
        state_update(site_id, status="failed", last_error="No URLs found.")
        raise RuntimeError("No URLs found. Check seed_url / allowed_paths.")

    cursor = int(st.get("cursor", 0))
    cursor = min(max(cursor, 0), total)

    print(f"[ingest] site_id={site_id} total={total} cursor={cursor}")
    print(f"[ingest] seed_url={seed_url}")
    print(f"[ingest] allowed_paths={allowed_paths} max_pages={max_pages} batch={batch_size}")

    while cursor < total:
        batch_urls = urls[cursor: cursor + batch_size]
        print(f"\n[batch] cursor={cursor} -> {cursor + len(batch_urls) - 1}")

        docs = []
        for u in batch_urls:
            try:
                state_update(site_id, last_url=u)
                r = session.get(u, timeout=TIMEOUT)
                if r.status_code != 200:
                    print(f"  - skip {u} status={r.status_code}")
                    continue
                if "text/html" not in r.headers.get("Content-Type", ""):
                    print(f"  - skip {u} content-type={r.headers.get('Content-Type')}")
                    continue

                title, text = extract_text(r.text)
                if len(text) < 50:
                    print(f"  - skip {u} (too short)")
                    continue

                chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
                docs.append((u, title, chunks))
                print(f"  + ok {u} chunks={len(chunks)}")

            except Exception as e:
                print(f"  ! error {u}: {e}")
                state_update(site_id, last_error=str(e))
                continue

        rows = []
        embed_inputs = []
        meta = []

        for (u, title, chunks) in docs:
            for i, c in enumerate(chunks):
                h = sha1(c)
                embed_inputs.append(c)
                meta.append((u, title, i, c, h))

        if embed_inputs:
            if dry_run:
                print(f"[dry-run] would embed {len(embed_inputs)} chunks")
            else:
                vectors = embed_texts(embed_inputs)
                for (u, title, idx, c, h), v in zip(meta, vectors):
                    rows.append({
                        "site_id": site_id,
                        "url": u,
                        "chunk_index": idx,
                        "title": title,
                        "content": c,
                        "embedding": v,
                        "updated_at": "now()",
                    })

                upsert_documents(rows)
                print(f"[db] upserted {len(rows)} chunks")

        cursor += batch_size
        state_update(site_id, cursor=min(cursor, total))

        if cursor < total and sleep_sec > 0:
            time.sleep(sleep_sec)

    state_update(site_id, status="done", last_error=None)
    print("\n[ingest] DONE")


# ==============
# sites.yml 読み込み（複数サイト）
# ==============
def load_sites_yml(path: str) -> list[dict]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --sites-yml. Please add pyyaml to requirements.txt") from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # 許容形式：
    # sites: [{site_id: 1, seed_url: "...", allowed_paths: [...]}, ...]
    sites = data.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("sites.yml must have 'sites:' as a non-empty list")

    norm = []
    for i, s in enumerate(sites):
        if not isinstance(s, dict):
            raise ValueError(f"sites.yml: sites[{i}] must be a mapping")

        site_id = s.get("site_id")
        seed_url = s.get("seed_url")
        if site_id is None or seed_url is None:
            raise ValueError(f"sites.yml: sites[{i}] requires site_id and seed_url")

        allowed_paths = s.get("allowed_paths", [])
        if isinstance(allowed_paths, str):
            # "a,b,c" 形式も許可
            allowed_paths = [x.strip() for x in allowed_paths.split(",") if x.strip()]
        elif isinstance(allowed_paths, list):
            allowed_paths = [str(x).strip() for x in allowed_paths if str(x).strip()]
        else:
            allowed_paths = []

        norm.append({
            "site_id": int(site_id),
            "seed_url": str(seed_url),
            "allowed_paths": allowed_paths,
            # サイト別override（無ければ None）
            "max_pages": s.get("max_pages"),
            "batch_size": s.get("batch_size"),
            "sleep_sec": s.get("sleep_sec"),
            "max_chars": s.get("max_chars"),
            "overlap": s.get("overlap"),
            "resume_from": s.get("resume_from"),
        })
    return norm


def parse_args():
    p = argparse.ArgumentParser()

    # 複数サイト実行
    p.add_argument("--sites-yml", type=str, default="")

    # 単体実行（sites-yml が無い場合に必須）
    p.add_argument("--site-id", type=int)
    p.add_argument("--seed-url", type=str)

    # 共通オプション（sites-yml でも共通値として使う）
    p.add_argument("--max-pages", type=int, default=300)
    p.add_argument("--allowed-paths", type=str, default="")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--sleep-sec", type=float, default=1.0)
    p.add_argument("--max-chars", type=int, default=2500)
    p.add_argument("--overlap", type=int, default=250)
    p.add_argument("--resume-from", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()

    # sites.yml モード
    if a.sites_yml:
        sites = load_sites_yml(a.sites_yml)

        # 共通 allowed_paths（CLIから渡したい場合）
        allowed_common = [s.strip() for s in a.allowed_paths.split(",") if s.strip()]

        print(f"[ingest] sites_yml={a.sites_yml} sites={len(sites)}")

        for s in sites:
            # サイト別指定があればそちら優先、なければ共通値
            site_id = s["site_id"]
            seed_url = s["seed_url"]

            allowed_paths = s["allowed_paths"] if s["allowed_paths"] else allowed_common

            max_pages = int(s["max_pages"]) if s["max_pages"] is not None else int(a.max_pages)
            batch_size = int(s["batch_size"]) if s["batch_size"] is not None else int(a.batch_size)
            sleep_sec = float(s["sleep_sec"]) if s["sleep_sec"] is not None else float(a.sleep_sec)
            max_chars = int(s["max_chars"]) if s["max_chars"] is not None else int(a.max_chars)
            overlap = int(s["overlap"]) if s["overlap"] is not None else int(a.overlap)
            resume_from = int(s["resume_from"]) if s["resume_from"] is not None else a.resume_from

            print("\n==============================")
            print(f"[site] site_id={site_id}")
            print(f"[site] seed_url={seed_url}")
            print(f"[site] allowed_paths={allowed_paths}")
            print(f"[site] max_pages={max_pages} batch_size={batch_size}")
            print("==============================")

            run_ingest(
                site_id=site_id,
                seed_url=seed_url,
                max_pages=max_pages,
                allowed_paths=allowed_paths,
                batch_size=batch_size,
                sleep_sec=sleep_sec,
                max_chars=max_chars,
                overlap=overlap,
                resume_from=resume_from,
                dry_run=a.dry_run,
            )

        print("\n[ingest] ALL SITES DONE")
        raise SystemExit(0)

    # 単体モード（従来互換）
    if a.site_id is None or a.seed_url is None:
        raise SystemExit("error: --site-id and --seed-url are required unless --sites-yml is provided")

    allowed = [s.strip() for s in a.allowed_paths.split(",") if s.strip()]
    run_ingest(
        site_id=int(a.site_id),
        seed_url=str(a.seed_url),
        max_pages=int(a.max_pages),
        allowed_paths=allowed,
        batch_size=int(a.batch_size),
        sleep_sec=float(a.sleep_sec),
        max_chars=int(a.max_chars),
        overlap=int(a.overlap),
        resume_from=a.resume_from,
        dry_run=a.dry_run,
    )
