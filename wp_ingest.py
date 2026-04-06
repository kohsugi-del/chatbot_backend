"""
WordPress REST API 差分取り込みスクリプト

- sites.yml の type: wordpress なサイトを対象
- modified_after で更新ページのみ取得（全クロール不要）
- 削除されたページを Supabase から自動削除
- 前回実行時刻を wp_last_run.json に保存（GitHub Actions でコミットして永続化）
"""

import os
import re
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import yaml

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from openai import OpenAI
from supabase import create_client, Client

log = logging.getLogger("wp_ingest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_CHARS = 1200
OVERLAP = 100

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (compatible; QwestIngestBot/1.0)"

# 前回実行時刻の保存先（GitHub Actions がコミットして永続化）
LAST_RUN_FILE = BASE_DIR / "wp_last_run.json"


# =============================
# ユーティリティ
# =============================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    chunks, i = [], 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        c = text[i:j].strip()
        if c:
            chunks.append(c)
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    res = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]


# =============================
# last_run 管理（ローカルファイル）
# =============================

def get_wp_last_run(site_id: int) -> Optional[str]:
    """前回実行時刻を取得。なければ None（初回は 7 日前をデフォルト）"""
    try:
        data = json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))
        return data.get(str(site_id))
    except Exception:
        return None


def set_wp_last_run(site_id: int, run_at: str):
    try:
        data: dict = {}
        if LAST_RUN_FILE.exists():
            data = json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))
        data[str(site_id)] = run_at
        LAST_RUN_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.warning(f"set_wp_last_run failed: {e}")


# =============================
# WordPress REST API 取得
# =============================

def fetch_wp_all_urls(base_url: str, post_type: str) -> set[str]:
    """削除検知用：WP 上の現在の全 URL を取得（id と link のみ）"""
    urls: set[str] = set()
    page = 1
    endpoint = f"{base_url.rstrip('/')}/wp-json/wp/v2/{post_type}"
    while True:
        try:
            r = session.get(endpoint, params={
                "per_page": 100, "page": page,
                "_fields": "id,link", "status": "publish",
            }, timeout=15)
            if r.status_code in (400, 404):
                break
            r.raise_for_status()
            items = r.json()
            if not items:
                break
            for item in items:
                link = (item.get("link") or "").rstrip("/") + "/"
                if link:
                    urls.add(link)
            if len(items) < 100:
                break
            page += 1
        except Exception as e:
            log.warning(f"fetch_wp_all_urls error ({post_type} page={page}): {e}")
            break
    return urls


def fetch_wp_updated(base_url: str, post_type: str, modified_after: str) -> list[dict]:
    """modified_after 以降に更新されたアイテムを全件取得"""
    items: list[dict] = []
    page = 1
    endpoint = f"{base_url.rstrip('/')}/wp-json/wp/v2/{post_type}"
    while True:
        try:
            r = session.get(endpoint, params={
                "per_page": 100, "page": page,
                "modified_after": modified_after,
                "status": "publish",
                "orderby": "modified",
                "order": "desc",
            }, timeout=15)
            if r.status_code in (400, 404):
                break
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        except Exception as e:
            log.warning(f"fetch_wp_updated error ({post_type} page={page}): {e}")
            break
    return items


# =============================
# Supabase: documents 操作
# =============================

def get_existing_urls(site_id: int) -> set[str]:
    """Supabase に保存済みの URL 一覧を取得"""
    urls: set[str] = set()
    try:
        r = (supabase.table("documents")
             .select("url")
             .eq("site_id", site_id)
             .execute())
        for row in (r.data or []):
            if row.get("url"):
                urls.add(row["url"])
    except Exception as e:
        log.warning(f"get_existing_urls error: {e}")
    return urls


def delete_documents_by_url(site_id: int, url: str):
    try:
        supabase.table("documents").delete().eq("site_id", site_id).eq("url", url).execute()
        log.info(f"  [delete] {url}")
    except Exception as e:
        log.warning(f"delete error {url}: {e}")


def upsert_documents(rows: list[dict]):
    supabase.table("documents").upsert(rows, on_conflict="site_id,url,chunk_index").execute()


# =============================
# メイン処理（サイト 1 件）
# =============================

def ingest_wp_site(cfg: dict, dry_run: bool = False) -> dict:
    site_id       = int(cfg["site_id"])
    base_url      = cfg["seed_url"].rstrip("/")
    post_types    = cfg.get("post_types", ["posts", "pages"])
    batch_size    = int(cfg.get("batch_size", 10))
    exclude_slugs = set(cfg.get("exclude_slugs", []))

    # ① 前回実行時刻（なければ 7 日前）
    last_run = get_wp_last_run(site_id)
    if last_run:
        modified_after = last_run
        log.info(f"[wp] site_id={site_id} modified_after={modified_after}")
    else:
        modified_after = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        log.info(f"[wp] site_id={site_id} 初回実行 → modified_after={modified_after}")

    run_start = now_iso()
    total_updated = 0
    total_deleted = 0

    for post_type in post_types:
        log.info(f"[wp] post_type={post_type}")

        # ② 更新ページを取得
        updated_items = fetch_wp_updated(base_url, post_type, modified_after)
        log.info(f"  updated={len(updated_items)}")

        # ③ 削除検知（WP 上の全 URL と DB の URL を比較）
        wp_urls  = fetch_wp_all_urls(base_url, post_type)
        db_urls  = get_existing_urls(site_id)
        # このサイト & post_type に属する URL のみ削除対象にする
        deleted_urls = {
            u for u in db_urls
            if u.startswith(base_url) and u not in wp_urls
        }
        if deleted_urls:
            log.info(f"  削除対象={len(deleted_urls)}")
            for url in deleted_urls:
                if not dry_run:
                    delete_documents_by_url(site_id, url)
                total_deleted += 1

        if not updated_items:
            log.info(f"  更新なし → スキップ")
            continue

        # ④ チャンク化・Embedding・Upsert
        for i in range(0, len(updated_items), batch_size):
            batch = updated_items[i : i + batch_size]
            docs = []

            for item in batch:
                url   = (item.get("link") or "").rstrip("/") + "/"
                slug  = item.get("slug", "")

                # exclude_slugs に一致したらスキップ
                if slug and slug in exclude_slugs:
                    log.info(f"  - skip (excluded slug) {url}")
                    continue

                title = strip_html(item.get("title", {}).get("rendered", ""))
                body  = strip_html(item.get("content", {}).get("rendered", ""))
                text  = f"{title}\n\n{body}".strip()

                if len(text) < 50:
                    log.info(f"  - skip (short) {url}")
                    continue

                chunks = chunk_text(text)
                docs.append((url, title, chunks))
                log.info(f"  + {url} chunks={len(chunks)}")

            if not docs:
                continue

            if dry_run:
                log.info(f"  [dry-run] would embed {sum(len(c) for _, _, c in docs)} chunks")
                continue

            embed_inputs = [c for _, _, chunks in docs for c in chunks]
            vectors = embed_texts(embed_inputs)

            rows = []
            vi = 0
            for (url, title, chunks) in docs:
                for idx, chunk in enumerate(chunks):
                    rows.append({
                        "site_id":     site_id,
                        "url":         url,
                        "chunk_index": idx,
                        "title":       title,
                        "content":     chunk,
                        "embedding":   vectors[vi],
                        "updated_at":  now_iso(),
                    })
                    vi += 1

            upsert_documents(rows)
            total_updated += len(docs)
            log.info(f"  [db] upserted {len(rows)} chunks")
            time.sleep(0.5)

    # ⑤ last_run を更新
    if not dry_run:
        set_wp_last_run(site_id, run_start)

    log.info(f"[wp] done site_id={site_id} updated={total_updated} deleted={total_deleted}")
    return {"site_id": site_id, "updated": total_updated, "deleted": total_deleted}


# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser(description="WordPress REST API 差分取り込み")
    parser.add_argument("--sites-yml", required=True, help="sites.yml のパス")
    parser.add_argument("--dry-run",   action="store_true", help="DB 書き込みをスキップ")
    parser.add_argument("--site-id",   type=int, default=None, help="特定 site_id のみ実行")
    args = parser.parse_args()

    with open(args.sites_yml, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    wp_sites = [s for s in config.get("sites", []) if s.get("type") == "wordpress"]
    if not wp_sites:
        log.info("sites.yml に type: wordpress のサイトがありません")
        return

    for cfg in wp_sites:
        if args.site_id and cfg["site_id"] != args.site_id:
            continue
        try:
            ingest_wp_site(cfg, dry_run=args.dry_run)
        except Exception:
            log.exception(f"site_id={cfg['site_id']} で予期しないエラー")


if __name__ == "__main__":
    main()
