# ingest.py
import os, re, time, hashlib, argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse
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
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def normalize_url(u: str) -> str:
    # フラグメントだけ落とす（クエリは残す）
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
# Crawl: sitemap優先 → 無ければ簡易BFS
# ==============
def fetch_sitemap_urls(seed_url: str, allowed_paths: list[str], max_pages: int) -> list[str]:
    parsed = urlparse(seed_url)
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    try:
        r = session.get(sitemap_url, timeout=TIMEOUT)
        r.encoding = r.apparent_encoding
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

        # まず取得してリンク抽出（許可パス外でもOK）
        try:
            r = session.get(u, timeout=TIMEOUT)
            r.encoding = r.apparent_encoding  # ★文字化け防止
            if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
                for link in extract_links(r.text, u):
                    lp = urlparse(link)
                    if lp.netloc == base_host and link not in seen:
                        q.append(link)
        except Exception:
            pass

        # documentsに入れるのは allowed のみ
        if is_allowed(u, base_host, allowed_paths):
            out.append(u)

    return out


# ==============
# DB（sitesテーブル）操作：status更新
# ==============
def db_engine_from_env():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required for --from-db mode (and for updating sites.status).")
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, pool_pre_ping=True, connect_args=connect_args)

def set_site_status(
    engine,
    site_id: int,
    status: str,
    last_error: str | None = None,
    ingested_urls: int | None = None,
):
    """
    sites テーブルに status / last_error / ingested_urls を書く。
    カラムが無い可能性があるので、できるだけ安全に更新する。
    """
    status = str(status)

    # まず status だけ必ず更新
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE sites SET status=:st WHERE id=:id"),
            {"st": status, "id": site_id},
        )

    # last_error 追記（あれば）
    if last_error is not None:
        with engine.begin() as conn:
            try:
                conn.execute(
                    text("UPDATE sites SET last_error=:err WHERE id=:id"),
                    {"err": (last_error[:2000] if last_error else None), "id": site_id},
                )
            except Exception:
                pass

    # ingested_urls 追記（あれば）
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
        q = text("SELECT id, url, scope, type, status FROM sites WHERE status='pending' ORDER BY id ASC LIMIT :lim")
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
    """
    戻り値:
      {
        "site_id": int,
        "total_urls": int,
        "ingested_urls": int,  # 実際に本文抽出できたURL数
        "chunks_upserted": int,
      }
    """
    st = state_get(site_id)

    if resume_from is not None:
        st["cursor"] = resume_from

    state_update(site_id, status="running", last_error=None)

    # URLs決定
    if urls_override is not None:
        urls = urls_override
    else:
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

    ingested_urls_count = 0
    chunks_upserted_total = 0

    while cursor < total:
        batch_urls = urls[cursor: cursor + batch_size]
        print(f"\n[batch] cursor={cursor} -> {cursor + len(batch_urls) - 1}")

        docs = []
        for u in batch_urls:
            try:
                state_update(site_id, last_url=u)
                r = session.get(u, timeout=TIMEOUT)
                r.encoding = r.apparent_encoding  # ★文字化け防止

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
                if not chunks:
                    print(f"  - skip {u} (no chunks)")
                    continue

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
                        "updated_at": now_iso(),
                    })

                upsert_documents(rows)
                chunks_upserted_total += len(rows)

                # docs の件数 = “本文抽出できたURL数”
                ingested_urls_count += len(docs)

                print(f"[db] upserted {len(rows)} chunks (urls_ok={len(docs)})")

        cursor += batch_size
        state_update(site_id, cursor=min(cursor, total))

        if cursor < total and sleep_sec > 0:
            time.sleep(sleep_sec)

    state_update(site_id, status="done", last_error=None)
    print("\n[ingest] DONE")

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
    - single: 指定URL1件だけ（urls_overrideを使う）
    - subtree: seed_url の path 配下に制限
    - other: 制限なし
    """
    scope = (scope or "").lower().strip()
    pu = urlparse(seed_url)
    path = pu.path or "/"

    if scope == "single":
        return ([], 1, [normalize_url(seed_url)])

    if scope == "subtree":
        ap = path if path.endswith("/") else (path.rsplit("/", 1)[0] + "/")
        return ([ap], None, None)

    # default: no restriction
    return ([], None, None)


# ==============
# ★ FastAPI から呼べる入口（重要）
# ==============
def ingest_site_from_db(
    site_id: int,
    *,
    max_pages: int = 300,
    batch_size: int = 10,
    sleep_sec: float = 1.0,
    max_chars: int = 2500,
    overlap: int = 250,
    dry_run: bool = False,
) -> dict:
    """
    FastAPI から:
      from ingest import ingest_site_from_db
      ingest_site_from_db(site_id=1)
    のように呼べる。

    - sites テーブルから url/scope を読む
    - status を crawling にする（フロントと合わせる）
    - 完了時に done + ingested_urls を反映
    """
    eng = db_engine_from_env()

    rows = fetch_sites_from_db(eng, limit=1, only_site_id=site_id, pending_only=False)
    if not rows:
        raise RuntimeError(f"site_id={site_id} not found in sites table")

    r = rows[0]
    seed_url = str(r["url"])
    scope = r.get("scope")

    allowed_paths_from_scope, max_pages_override, urls_override = derive_allowed_from_scope(seed_url, scope)

    mp = int(max_pages_override) if max_pages_override is not None else int(max_pages)

    # ★ フロント想定の status に合わせる
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
            resume_from=0,  # API起動は基本リスタート
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

    # ★ DB（sitesテーブル）から pending を拾う
    p.add_argument("--from-db", action="store_true", help="Ingest pending sites from DB (sites table)")
    p.add_argument("--limit", type=int, default=20, help="How many pending sites to process in one run")
    p.add_argument("--only-site-id", type=int, default=None, help="Process only this site_id (from sites table)")

    # 複数サイト実行（従来）
    p.add_argument("--sites-yml", type=str, default="")

    # 単体実行（従来）
    p.add_argument("--site-id", type=int)
    p.add_argument("--seed-url", type=str)

    # 共通オプション
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
    # ★ DBモード：sitesテーブルの pending を処理
    # =========================
    if a.from_db:
        eng = db_engine_from_env()
        rows = fetch_sites_from_db(
            eng,
            limit=int(a.limit),
            only_site_id=a.only_site_id,
            pending_only=(a.only_site_id is None),  # only指定ならpending以外でも処理可能
        )

        print(f"[db] sites found: {len(rows)} (limit={a.limit})")

        for r in rows:
            site_id = int(r["id"])
            seed_url = str(r["url"])
            scope = r.get("scope")
            st = str(r.get("status") or "")

            # only指定がないときだけ pending 以外スキップ
            if a.only_site_id is None and st != "pending":
                continue

            try:
                print("\n==============================")
                print(f"[site] id={site_id} scope={scope} url={seed_url}")
                print("==============================")

                # ★ フロント想定の status に合わせる
                set_site_status(eng, site_id, "crawling", last_error=None)

                allowed_paths_from_scope, max_pages_override, urls_override = derive_allowed_from_scope(seed_url, scope)

                # CLI allowed_paths が指定されていれば優先（従来互換）
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
                    resume_from=0,   # ★ DBモードは毎回先頭から
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
                print(f"[site] ERROR site_id={site_id}: {err}")
                set_site_status(eng, site_id, "error", last_error=err)

        print("\n[db] ALL DONE")
        raise SystemExit(0)

    # =========================
    # sites.yml モード（従来）
    # =========================
    if a.sites_yml:
        sites = load_sites_yml(a.sites_yml)

        allowed_common = [s.strip() for s in a.allowed_paths.split(",") if s.strip()]
        print(f"[ingest] sites_yml={a.sites_yml} sites={len(sites)}")

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

    # =========================
    # 単体モード（従来互換）
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
