# ingest.py
import os, re, time, hashlib, argparse, logging
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urlunparse,
    parse_qsl,
    urlencode,
)
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ★ .env をこのファイルの隣から必ず読む（起動場所ズレ対策）
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# OpenAI（新SDK）
from openai import OpenAI

# Supabase
from supabase import create_client, Client

# SQLAlchemy（DBの sites テーブルを読む）
from sqlalchemy import create_engine, text


# ==============
# 設定
# ==============
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))

# ★ connect/read を分離（環境変数が無い場合は TIMEOUT から決める）
TIMEOUT_CONNECT = int(os.getenv("HTTP_TIMEOUT_CONNECT", str(min(5, max(1, TIMEOUT // 4)))))
TIMEOUT_READ = int(os.getenv("HTTP_TIMEOUT_READ", str(max(8, TIMEOUT))))

UA = os.getenv(
    "INGEST_UA",
    "Mozilla/5.0 (compatible; QwestIngestBot/1.0; +https://qwest.co.jp)"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing in .env")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY is missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)

session = requests.Session()
session.headers.update({"User-Agent": UA})

# ★ FastAPI(uvicorn) のログに寄せる（BackgroundTasksでも追いやすい）
log = logging.getLogger("uvicorn")
log.setLevel(logging.INFO)

# ★ 単体実行（python ingest.py）でもログが出るように保険
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ===== 共通フィルタ設定 =====
DENY_PATTERNS = [
    r"/page/\d+/?$",   # ページネーション
    r"/tag/",
    r"/category/",
    r"/wp-content/",
    r"/wp-json/",
]
DENY_QUERY = True      # 既定は ? 付きURLを除外（※ /plus/ 配下のみ例外で許可）
DENY_FRAGMENT = True   # #付きURLは除外


# ★ クエリを残したい場合は許可キーを追加
# /plus/ 配下は query が実体なので、ここが超重要
ALLOW_QUERY_KEYS = {"page", "app_controller", "id", "type", "run"}


# ==============
# ユーティリティ
# ==============
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _norm_host(h: str) -> str:
    """www差を無視してホスト比較できるようにする"""
    h = (h or "").strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h

def _looks_like_file(path: str) -> bool:
    """末尾が .php .html .pdf などの “ファイル” に見えるか"""
    return bool(re.search(r"\.[a-zA-Z0-9]{1,6}$", path or ""))

def ensure_plus_search_run(u: str) -> str:
    """/plus/ search URL を run=true 付きに正規化する"""
    p = urlparse(u)
    if not (p.path or "").startswith("/plus/"):
        return u

    qd = dict(parse_qsl(p.query, keep_blank_values=True))
    if qd.get("app_controller") != "search":
        return u

    # run=true を強制
    qd["run"] = "true"

    # なるべく無限増殖を避けたいので、page が無ければ 1 を入れる（任意）
    # qd.setdefault("page", "1")

    q = urlencode(sorted(qd.items()), doseq=True)
    return normalize_url(urlunparse((p.scheme, p.netloc, p.path, p.params, q, "")))

def normalize_url(u: str) -> str:
    """
    URLの表記ゆれを統一して、visited/重複判定の精度を上げる。
    - fragment は必ず削除
    - query は原則削除（ALLOW_QUERY_KEYSだけ残す）
    - path 末尾スラッシュを統一（ただし .php 等の “ファイル” は付けない）
    """
    u = (u or "").strip()
    if not u:
        return u

    p = urlparse(u)

    # fragment は必ず落とす
    frag = ""

    # query は許可キーだけ残す
    q = ""
    if p.query:
        pairs = [
            (k, v)
            for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k in ALLOW_QUERY_KEYS
        ]
        if pairs:
            q = urlencode(pairs, doseq=True)

    # path を統一
    path = p.path or "/"
    if path != "/" and (not _looks_like_file(path)) and (not path.endswith("/")):
        path += "/"

    return urlunparse((p.scheme, p.netloc, path, p.params, q, frag))

def _norm_path(path: str) -> str:
    """allowed_paths 側も正規化（ファイルっぽいものは末尾/を付けない）"""
    path = path or "/"
    if path != "/" and (not _looks_like_file(path)) and (not path.endswith("/")):
        path += "/"
    return path

def is_allowed(url: str, base_host: str, allowed_paths: list[str]) -> bool:
    """
    allowed_paths とURLの末尾 / 揺れで落ちないように、両方正規化して比較する。
    - ★ /plus/ 配下は query を許可（ただし info&id のみ許可）
    """
    p = urlparse(url)

    # ドメインチェック（★www差を無視）
    if _norm_host(p.netloc) != _norm_host(base_host):
        return False

    # フラグメント除外
    if DENY_FRAGMENT and p.fragment:
        return False

    # ★ /plus/ 配下は query を許可、それ以外はDENY_QUERYなら落とす
    if DENY_QUERY and p.query and (not (p.path or "").startswith("/plus/")):
        return False

     # ★ /plus/ は info&id のみ許可（クエリ無しの /plus/ や login.php は除外）
    if (p.path or "").startswith("/plus/"):
        # loginは除外
        if (p.path or "").endswith("/plus/login.php"):
            return False

        # queryが無い /plus/ 系は除外（入口ページまで入れたいなら True にしてもOK）
        if not p.query:
            return False

        qd = dict(parse_qsl(p.query, keep_blank_values=True))
        if qd.get("app_controller") != "info":
            return False
        if "id" not in qd:
            return False

    path = _norm_path(p.path)

    # deny パターン
    for pat in DENY_PATTERNS:
        if re.search(pat, path):
            return False

    # allow 判定
    if not allowed_paths:
        return True

    allowed_norm = [_norm_path(str(ap).strip()) for ap in allowed_paths if ap and str(ap).strip()]
    return any(path.startswith(ap) for ap in allowed_norm)

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        absu = urljoin(base_url, href)
        absu = normalize_url(absu)
        if absu:
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

def normalize_text_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def make_page_hash(title: str, text: str) -> str:
    base = (title or "") + "\n" + normalize_text_for_hash(text)
    return sha1(base)

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
        "updated_at": now_iso(),
    }
    supabase.table("ingest_state").insert(init).execute()
    return init

def state_update(site_id: int, **kwargs):
    kwargs["updated_at"] = now_iso()
    supabase.table("ingest_state").update(kwargs).eq("site_id", site_id).execute()


# ==============
# Supabase: documents upsert
# ==============
def upsert_documents(rows: list[dict]):
    supabase.table("documents").upsert(rows, on_conflict="site_id,url,chunk_index").execute()


# ==============
# Supabase: page fingerprints（★同一内容の再取り込み防止）
# ==============
def fingerprint_get(site_id: int, url: str) -> str | None:
    try:
        r = (
            supabase.table("page_fingerprints")
            .select("page_hash")
            .eq("site_id", site_id)
            .eq("url", url)
            .execute()
        )
        if r.data:
            return r.data[0].get("page_hash")
    except Exception:
        return None
    return None

def fingerprint_upsert(site_id: int, url: str, page_hash: str):
    try:
        supabase.table("page_fingerprints").upsert(
            {
                "site_id": site_id,
                "url": url,
                "page_hash": page_hash,
                "updated_at": now_iso(),
            },
            on_conflict="site_id,url",
        ).execute()
    except Exception:
        pass


# ==============
# Crawl: sitemap優先 → 無ければ簡易BFS
# ==============
def fetch_sitemap_urls(seed_url: str, allowed_paths: list[str], max_pages: int) -> list[str]:
    parsed = urlparse(seed_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    base_host = parsed.netloc

    candidates = [
        f"{base}/sitemap.xml",
        f"{base}/sitemap_index.xml",
        f"{base}/wp-sitemap.xml",
    ]

    collected: list[str] = []
    seen: set[str] = set()

    def add_url(u: str):
        nonlocal collected
        u = normalize_url(u)
        if not u or u in seen:
            return
        if is_allowed(u, base_host, allowed_paths):
            seen.add(u)
            collected.append(u)

    def parse_urlset(xml_text: str):
        soup = BeautifulSoup(xml_text, "xml")
        for loc in soup.select("url > loc"):
            add_url(loc.get_text(strip=True))
            if len(collected) >= max_pages:
                break

    def parse_sitemapindex(xml_text: str):
        soup = BeautifulSoup(xml_text, "xml")
        locs = [x.get_text(strip=True) for x in soup.select("sitemap > loc")]
        for sm in locs:
            if len(collected) >= max_pages:
                break
            try:
                rr = session.get(sm, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
                rr.encoding = rr.apparent_encoding
                if rr.status_code != 200:
                    continue
                text_ = rr.text
                if "<sitemapindex" in text_:
                    parse_sitemapindex(text_)
                if "<urlset" in text_:
                    parse_urlset(text_)
            except requests.exceptions.RequestException:
                continue

    for sm_url in candidates:
        try:
            r = session.get(sm_url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
            r.encoding = r.apparent_encoding
            if r.status_code != 200:
                continue

            text_ = r.text
            if "<sitemapindex" in text_:
                parse_sitemapindex(text_)
                if collected:
                    return collected[:max_pages]
            if "<urlset" in text_:
                parse_urlset(text_)
                if collected:
                    return collected[:max_pages]
        except requests.exceptions.RequestException:
            continue

def _is_plus_path(u: str) -> bool:
    return (urlparse(u).path or "").startswith("/plus/")

def _is_plus_search(u: str) -> bool:
    p = urlparse(u)
    if not (p.path or "").startswith("/plus/"):
        return False
    qd = dict(parse_qsl(p.query, keep_blank_values=True))
    return qd.get("app_controller") == "search"

def _is_plus_info(u: str) -> bool:
    p = urlparse(u)
    if not (p.path or "").startswith("/plus/"):
        return False
    qd = dict(parse_qsl(p.query, keep_blank_values=True))
    return qd.get("app_controller") == "info" and ("id" in qd)

def bfs_crawl(seed_url: str, allowed_paths: list[str], max_pages: int) -> list[str]:
    parsed = urlparse(seed_url)
    base_host = parsed.netloc

    # ★ 追加：seed が /plus/ かどうか（1回だけ判定して使い回す）
    seed_is_plus = (urlparse(seed_url).path or "").startswith("/plus/")

    q = [normalize_url(seed_url)]
    seen: set[str] = set()
    out: list[str] = []

    while q and len(out) < max_pages:
        u = q.pop(0)
        if not u or u in seen:
            continue
        seen.add(u)

        try:
            log.info(f"[bfs][GET] {u}")
            r = session.get(u, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
            r.encoding = r.apparent_encoding
            ct = (r.headers.get("Content-Type") or "").lower()

            if r.status_code == 200 and (("text/html" in ct) or ("application/xhtml+xml" in ct)):
                links = extract_links(r.text, u)

                # =========================
                # ★ /plus/ の挙動を特殊化
                # - search は「中身を見るだけ」
                # - search からは info だけをキューに積む
                # - search 自体は out に入れない
                # =========================
                if _is_plus_search(u):
                    for link in links:
                        lp = urlparse(link)

                        # 同一ドメインのみ
                        if _norm_host(lp.netloc) != _norm_host(base_host):
                            continue
                        if link in seen:
                            continue

                        # ★ 追加：seed が /plus/ の時は /plus/ 外へ脱線しない
                        if seed_is_plus and not (lp.path or "").startswith("/plus/"):
                            continue

                        # searchページは増殖させない（page=2 等に行かない）
                        if _is_plus_search(link):
                            # search は増殖しやすいので「run=true の1種類」だけ許す
                            q.append(ensure_plus_search_run(link))
                            continue

                        # info だけ許可してキューへ
                        if _is_plus_info(link):
                            q.append(link)

                    # search は out に入れない
                    continue

                # =========================
                # 通常の挙動：辿るURLも is_allowed で絞る
                # =========================
                for link in links:
                    lp = urlparse(link)

                    # 同一ドメインのみ
                    if _norm_host(lp.netloc) != _norm_host(base_host):
                        continue
                    if link in seen:
                        continue

                    # ★ 追加：seed が /plus/ の時は /plus/ 外へ脱線しない
                    if seed_is_plus and not (lp.path or "").startswith("/plus/"):
                        continue

                    # /plus/ search は「見るだけ」なので、キューには積む（1回だけ）
                    if _is_plus_search(link):
                        q.append(ensure_plus_search_run(link))
                        continue

                    # それ以外は is_allowed を通す
                    if not is_allowed(link, base_host, allowed_paths):
                        continue

                    q.append(link)

        except requests.exceptions.RequestException as e:
            log.info(f"  - bfs skip {u} ({type(e).__name__})")
            continue

        # documentsに入れるのは allowed のみ
        if is_allowed(u, base_host, allowed_paths):
            out.append(u)

    return out



# ==============
# DB（sitesテーブル）操作：status更新
# ==============
def db_engine_from_env():
    from database import engine
    return engine

def set_site_status(
    engine,
    site_id: int,
    status: str,
    last_error: str | None = None,
    ingested_urls: int | None = None,
):
    status = str(status)

    with engine.begin() as conn:
        conn.execute(
            text("UPDATE sites SET status=:st WHERE id=:id"),
            {"st": status, "id": site_id},
        )

    if last_error is not None:
        with engine.begin() as conn:
            try:
                conn.execute(
                    text("UPDATE sites SET last_error=:err WHERE id=:id"),
                    {"err": (last_error[:2000] if last_error else None), "id": site_id},
                )
            except Exception:
                pass

    if ingested_urls is not None:
        with engine.begin() as conn:
            try:
                conn.execute(
                    text("UPDATE sites SET ingested_urls=:n WHERE id=:id"),
                    {"n": int(ingested_urls), "id": site_id},
                )
            except Exception:
                pass

def fetch_sites_from_db(engine, limit: int, only_site_id: int | None, pending_only: bool):
    if only_site_id is not None:
        q = text("SELECT id, url, scope, type, status FROM sites WHERE id=:id LIMIT 1")
        with engine.connect() as conn:
            row = conn.execute(q, {"id": only_site_id}).mappings().first()
        return [row] if row else []

    if pending_only:
        q = text(
            "SELECT id, url, scope, type, status "
            "FROM sites WHERE status='pending' ORDER BY id ASC LIMIT :lim"
        )
    else:
        q = text("SELECT id, url, scope, type, status FROM sites ORDER BY id ASC LIMIT :lim")

    with engine.connect() as conn:
        rows = conn.execute(q, {"lim": limit}).mappings().all()
    return list(rows)


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
    urls_override: list[str] | None = None,
) -> dict:
    st = state_get(site_id)

    if resume_from is not None:
        st["cursor"] = resume_from

    state_update(site_id, status="running", last_error=None)

    # ★ /plus/ の時は探索・取り込み対象を /plus/ 配下に固定（脱線防止）
    if (urlparse(seed_url).path or "").startswith("/plus/"):
        if not allowed_paths:
            allowed_paths = ["/plus/"]
            log.info("[plus] force allowed_paths=['/plus/']")

    # URLs決定
    if urls_override is not None:
        urls = [normalize_url(u) for u in urls_override if normalize_url(u)]
        crawl_mode = "override"
    else:
        # ★ /plus/ は sitemap が役に立たない（クエリ詳細が載らない）ので BFS 強制
        if (urlparse(seed_url).path or "").startswith("/plus/"):
            crawl_mode = "bfs"
            urls = bfs_crawl(seed_url, allowed_paths, max_pages)
        else:
            urls = fetch_sitemap_urls(seed_url, allowed_paths, max_pages)
            if urls:
                crawl_mode = "sitemap"
            else:
                crawl_mode = "bfs"
                urls = bfs_crawl(seed_url, allowed_paths, max_pages)

    total = len(urls)
    state_update(site_id, total=total)

    # ★ dry-runでも「収集できてるか」必ず見えるように
    log.info(f"[crawl] mode={crawl_mode} collected={total} (max_pages={max_pages})")

    if total == 0:
        state_update(site_id, status="failed", last_error="No URLs found.")
        if (urlparse(seed_url).path or "").startswith("/plus/"):
            log.info("[plus] No URLs found (info links not detected). End without raising.")
            return {
                "site_id": site_id,
                "total_urls": 0,
                "ingested_urls": 0,
                "chunks_upserted": 0,
            }
        raise RuntimeError("No URLs found. Check seed_url / allowed_paths.")

    cursor = int(st.get("cursor", 0))
    cursor = min(max(cursor, 0), total)

    log.info(f"[ingest] site_id={site_id} total={total} cursor={cursor}")
    log.info(f"[ingest] seed_url={seed_url}")
    log.info(f"[ingest] allowed_paths={allowed_paths} batch={batch_size}")
    log.info(f"[ingest] timeout(connect,read)=({TIMEOUT_CONNECT},{TIMEOUT_READ}) dry_run={dry_run}")

    ingested_urls_count = 0
    chunks_upserted_total = 0

    while cursor < total:
        batch_urls = urls[cursor: cursor + batch_size]
        log.info(f"[batch] cursor={cursor} -> {cursor + len(batch_urls) - 1}")

        docs = []
        for u in batch_urls:
            try:
                u = normalize_url(u)
                if not u:
                    continue

                state_update(site_id, last_url=u)

                # ★ どこで止まりやすいか分かるように（必要ならコメントアウトOK）
                log.info(f"  [GET] {u}")

                r = session.get(u, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
                r.encoding = r.apparent_encoding

                if r.status_code != 200:
                    log.info(f"  - skip {u} status={r.status_code}")
                    continue

                ct = (r.headers.get("Content-Type") or "").lower()
                if ("text/html" not in ct) and ("application/xhtml+xml" not in ct):
                    log.info(f"  - skip {u} content-type={ct}")
                    continue

                title, text_ = extract_text(r.text)
                if len(text_) < 50:
                    log.info(f"  - skip {u} (too short)")
                    continue

                page_hash = make_page_hash(title, text_)
                prev = fingerprint_get(site_id, u)
                if prev is not None and prev == page_hash:
                    log.info(f"  - skip {u} (unchanged)")
                    continue

                chunks = chunk_text(text_, max_chars=max_chars, overlap=overlap)
                if not chunks:
                    log.info(f"  - skip {u} (no chunks)")
                    continue

                docs.append((u, title, chunks, page_hash))
                log.info(f"  + ok {u} chunks={len(chunks)}")

            except Exception as e:
                log.exception(f"  ! error {u}: {e}")
                state_update(site_id, last_error=str(e))
                continue

        rows = []
        embed_inputs = []
        meta = []

        for (u, title, chunks, page_hash) in docs:
            for i, c in enumerate(chunks):
                h = sha1(c)
                embed_inputs.append(c)
                meta.append((u, title, i, c, h, page_hash))

        if embed_inputs:
            if dry_run:
                log.info(f"[dry-run] would embed {len(embed_inputs)} chunks (urls_ok={len(docs)})")
            else:
                vectors = embed_texts(embed_inputs)
                for (u, title, idx, c, h, page_hash), v in zip(meta, vectors):
                    rows.append({
                        "site_id": site_id,
                        "url": u,
                        "chunk_index": idx,
                        "title": title,
                        "content": c,
                        "embedding": v,
                        "updated_at": now_iso(),
                    })

                upsert_documents(rows)
                chunks_upserted_total += len(rows)
                ingested_urls_count += len(docs)

                for (u, _title, _chunks, _page_hash) in docs:
                    fingerprint_upsert(site_id, u, _page_hash)

                log.info(f"[db] upserted {len(rows)} chunks (urls_ok={len(docs)})")

        cursor += batch_size
        state_update(site_id, cursor=min(cursor, total))

        if cursor < total and sleep_sec > 0:
            time.sleep(sleep_sec)

    state_update(site_id, status="done", last_error=None)
    log.info("[ingest] DONE")

    return {
        "site_id": site_id,
        "total_urls": total,
        "ingested_urls": ingested_urls_count,
        "chunks_upserted": chunks_upserted_total,
    }


# ==============
# scope から制限を導出
# ==============
def derive_allowed_from_scope(seed_url: str, scope: str | None) -> tuple[list[str], int | None, list[str] | None]:
    """
    sites.scope から allowed_paths / max_pages / urls_override を決める
    - single: 指定URL1件だけ
    - subtree: seed_url の path 配下に制限
    - other: 制限なし
    """
    scope = (scope or "").lower().strip()
    pu = urlparse(seed_url)
    path = pu.path or "/"

    if scope == "single":
        return ([], 1, [normalize_url(seed_url)])

    if scope == "subtree":
        ap = _norm_path(path)
        return ([ap], None, None)

    return ([], None, None)


# ==============
# ★ FastAPI から呼べる入口（重要）
# ==============
def ingest_site_from_db(
    site_id: int,
    *,
    max_pages: int = 120,
    batch_size: int = 8,
    sleep_sec=0.1,
    max_chars=4500,
    overlap: int = 80,
    dry_run: bool = False,
) -> dict:
    eng = db_engine_from_env()

    rows = fetch_sites_from_db(eng, limit=1, only_site_id=site_id, pending_only=False)
    if not rows:
        raise RuntimeError(f"site_id={site_id} not found in sites table")

    r = rows[0]
    seed_url = str(r["url"])
    scope = r.get("scope")

    allowed_paths_from_scope, max_pages_override, urls_override = derive_allowed_from_scope(seed_url, scope)
    mp = int(max_pages_override) if max_pages_override is not None else int(max_pages)

    set_site_status(eng, site_id, "crawling", last_error=None)

    try:
        result = run_ingest(
            site_id=site_id,
            seed_url=seed_url,
            max_pages=mp,
            allowed_paths=allowed_paths_from_scope,
            batch_size=int(batch_size),
            sleep_sec=float(sleep_sec),
            max_chars=int(max_chars),
            overlap=int(overlap),
            resume_from=0,
            dry_run=bool(dry_run),
            urls_override=urls_override,
        )

        set_site_status(
            eng,
            site_id,
            "done",
            last_error=None,
            ingested_urls=int(result.get("ingested_urls", 0)),
        )
        return result

    except Exception as e:
        err = str(e)
        set_site_status(eng, site_id, "error", last_error=err)
        raise


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
            allowed_paths = [x.strip() for x in allowed_paths.split(",") if x.strip()]
        elif isinstance(allowed_paths, list):
            allowed_paths = [str(x).strip() for x in allowed_paths if str(x).strip()]
        else:
            allowed_paths = []

        norm.append({
            "site_id": int(site_id),
            "seed_url": str(seed_url),
            "allowed_paths": allowed_paths,
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

    p.add_argument("--from-db", action="store_true", help="Ingest pending sites from DB (sites table)")
    p.add_argument("--limit", type=int, default=20, help="How many pending sites to process in one run")
    p.add_argument("--only-site-id", type=int, default=None, help="Process only this site_id (from sites table)")

    p.add_argument("--sites-yml", type=str, default="")

    p.add_argument("--site-id", type=int)
    p.add_argument("--seed-url", type=str)

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

    # =========================
    # DBモード：sitesテーブルの pending を処理
    # =========================
    if a.from_db:
        eng = db_engine_from_env()
        rows = fetch_sites_from_db(
            eng,
            limit=int(a.limit),
            only_site_id=a.only_site_id,
            pending_only=(a.only_site_id is None),
        )

        log.info(f"[db] sites found: {len(rows)} (limit={a.limit})")

        for r in rows:
            site_id = int(r["id"])
            seed_url = str(r["url"])
            scope = r.get("scope")
            st = str(r.get("status") or "")

            if a.only_site_id is None and st != "pending":
                continue

            try:
                log.info("==============================")
                log.info(f"[site] id={site_id} scope={scope} url={seed_url}")
                log.info("==============================")

                set_site_status(eng, site_id, "crawling", last_error=None)

                allowed_paths_from_scope, max_pages_override, urls_override = derive_allowed_from_scope(seed_url, scope)

                allowed_cli = [s.strip() for s in a.allowed_paths.split(",") if s.strip()]
                allowed_paths = allowed_cli if allowed_cli else allowed_paths_from_scope

                max_pages = int(a.max_pages)
                if max_pages_override is not None:
                    max_pages = int(max_pages_override)

                result = run_ingest(
                    site_id=site_id,
                    seed_url=seed_url,
                    max_pages=max_pages,
                    allowed_paths=allowed_paths,
                    batch_size=int(a.batch_size),
                    sleep_sec=float(a.sleep_sec),
                    max_chars=int(a.max_chars),
                    overlap=int(a.overlap),
                    resume_from=0,
                    dry_run=a.dry_run,
                    urls_override=urls_override,
                )

                set_site_status(
                    eng,
                    site_id,
                    "done",
                    last_error=None,
                    ingested_urls=int(result.get("ingested_urls", 0)),
                )

            except Exception as e:
                err = str(e)
                log.exception(f"[site] ERROR site_id={site_id}: {err}")
                set_site_status(eng, site_id, "error", last_error=err)

        log.info("[db] ALL DONE")
        raise SystemExit(0)

    # =========================
    # sites.yml モード
    # =========================
    if a.sites_yml:
        sites = load_sites_yml(a.sites_yml)

        allowed_common = [s.strip() for s in a.allowed_paths.split(",") if s.strip()]
        log.info(f"[ingest] sites_yml={a.sites_yml} sites={len(sites)}")

        for s in sites:
            site_id = s["site_id"]
            seed_url = s["seed_url"]

            allowed_paths = s["allowed_paths"] if s["allowed_paths"] else allowed_common

            max_pages = int(s["max_pages"]) if s["max_pages"] is not None else int(a.max_pages)
            batch_size = int(s["batch_size"]) if s["batch_size"] is not None else int(a.batch_size)
            sleep_sec = float(s["sleep_sec"]) if s["sleep_sec"] is not None else float(a.sleep_sec)
            max_chars = int(s["max_chars"]) if s["max_chars"] is not None else int(a.max_chars)
            overlap = int(s["overlap"]) if s["overlap"] is not None else int(a.overlap)
            resume_from = int(s["resume_from"]) if s["resume_from"] is not None else a.resume_from

            log.info("==============================")
            log.info(f"[site] site_id={site_id}")
            log.info(f"[site] seed_url={seed_url}")
            log.info(f"[site] allowed_paths={allowed_paths}")
            log.info(f"[site] max_pages={max_pages} batch_size={batch_size}")
            log.info("==============================")

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

        log.info("[ingest] ALL SITES DONE")
        raise SystemExit(0)

    # =========================
    # 単体モード
    # =========================
    if a.site_id is None or a.seed_url is None:
        raise SystemExit("error: --site-id and --seed-url are required unless --sites-yml is provided or --from-db")

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
