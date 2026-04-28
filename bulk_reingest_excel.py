"""
AIチャットボット登録URL (1).xlsx の77件URLを一括再取り込みするスクリプト。

使い方:
    cd chatbot_backend
    python bulk_reingest_excel.py [--dry-run] [--skip-ingest]

オプション:
    --dry-run       DBへの登録・Supabaseへの書き込みは行わず動作確認のみ
    --skip-ingest   サイト登録だけ行い、取り込みは後で手動実行する
"""

import sys
import argparse
import logging
from pathlib import Path

# .env をこのファイルの隣から読む
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from database import SessionLocal, engine
from models_site import Site, Base

Base.metadata.create_all(bind=engine)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ===== Excelから抽出した77件のURL =====
EXCEL_URLS = [
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/d056429.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/d079372.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/d081618.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/d081602.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/d079421.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0004/d058999.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/p003956-5.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/p003956-6.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/p003953-2.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/p003956-9.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081451.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d079297.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0002/d059863.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/d054291.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/d081079.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d074125.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/d074170.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/p003953-3.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/d054343.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d064521.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d072059.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/d074031.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/p003953-7.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/d066017.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d078850.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0003/p003956-8.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0004/d054297.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0004/p003395.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0005/p003961.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0006/p003909.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0007/d077071.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/index.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081439.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081447.html",
    "https://www.city.asahikawa.hokkaido.jp/1200/d080841.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081130.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081437.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081434.html",
    "https://www.city.asahikawa.hokkaido.jp/1200/d076923.html",
    "https://www.city.asahikawa.hokkaido.jp/1400/ijyu/ijyu_update/d081435.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/135/160/170/d065872.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/135/160/171/d080874.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/135/160/171/d079907.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/135/160/171/d079986.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/135/160/171/d080219.html",
    "https://www.city.asahikawa.hokkaido.jp/500/548/koureisya/osirase/d080221.html",
    "https://www.city.asahikawa.hokkaido.jp/500/548/koureisya/osirase/d082803.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d082291.html",
    "https://www.city.asahikawa.hokkaido.jp/1000/1001/sonota/sonota/p003607.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d081607.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d072852.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d075903.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d073558.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d058510.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d072423.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d056508.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/p003956-7.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d082123.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d082076.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d079832.html",
    "https://www.city.asahikawa.hokkaido.jp/500/501/508/d082266.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d082144.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d080175.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d081878.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/408/600/d082111.html",
    "https://www.city.asahikawa.hokkaido.jp/700/735/751/752/d056836.html",
    "https://www.city.asahikawa.hokkaido.jp/500/501/508/d066823.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/452/459/460/d079599.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/452/453/454/d069541.html",
    "https://www.city.asahikawa.hokkaido.jp/500/548/kosodate/osirase/d064735.html",
    "https://www.city.asahikawa.hokkaido.jp/500/548/kosodate/osirase/d063894.html",
    "https://www.city.asahikawa.hokkaido.jp/500/501/504/d079585.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/374/378/p005484.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/374/376/d075365.html",
    "https://www.city.asahikawa.hokkaido.jp/10013/10014/d066834.html",
    "https://www.city.asahikawa.hokkaido.jp/500/562/work202/d080852.html",
    "https://www.city.asahikawa.hokkaido.jp/kurashi/364/392/work0001/d081450.html",
]


def ensure_sites(db, urls: list[str], dry_run: bool) -> list[int]:
    """DBに未登録のURLを追加し、全URLのsite_idリストを返す"""
    existing = {row.url: row.id for row in db.query(Site).all()}

    site_ids = []
    new_count = 0
    existing_count = 0

    for url in urls:
        if url in existing:
            log.info(f"[existing] id={existing[url]} {url}")
            site_ids.append(existing[url])
            existing_count += 1
        else:
            log.info(f"[new]      {url}")
            if not dry_run:
                site = Site(url=url, scope="single", type="静的HTML", status="pending")
                db.add(site)
                db.flush()
                site_ids.append(site.id)
            new_count += 1

    if not dry_run:
        db.commit()

    log.info(f"既存: {existing_count}件 / 新規登録: {new_count}件")
    return site_ids


def main():
    parser = argparse.ArgumentParser(description="Excel77件URL 一括再取り込み")
    parser.add_argument("--dry-run", action="store_true", help="DB/Supabase書き込みなし（動作確認）")
    parser.add_argument("--skip-ingest", action="store_true", help="サイト登録のみ・取り込みはスキップ")
    args = parser.parse_args()

    log.info(f"=== bulk_reingest_excel.py 開始 (dry_run={args.dry_run}, skip_ingest={args.skip_ingest}) ===")
    log.info(f"対象URL: {len(EXCEL_URLS)}件")

    db = SessionLocal()
    try:
        site_ids = ensure_sites(db, EXCEL_URLS, dry_run=args.dry_run)
    finally:
        db.close()

    if args.skip_ingest or args.dry_run:
        log.info("--skip-ingest / --dry-run のため取り込みをスキップします")
        log.info(f"site_ids: {site_ids}")
        return

    # 取り込み実行
    from ingest import ingest_site_from_db

    ok = 0
    ng = 0

    for i, site_id in enumerate(site_ids, 1):
        log.info(f"[{i}/{len(site_ids)}] ingest開始 site_id={site_id}")
        try:
            result = ingest_site_from_db(
                site_id,
                max_pages=1,       # scope=single なので1ページのみ
                batch_size=5,
                sleep_sec=0.2,
                dry_run=False,
            )
            log.info(f"  -> 完了 ingested_urls={result.get('ingested_urls')} chunks={result.get('chunks_upserted')}")
            ok += 1
        except Exception as e:
            log.error(f"  -> エラー site_id={site_id}: {e}")
            ng += 1

    log.info(f"=== 完了: 成功 {ok}件 / 失敗 {ng}件 ===")


if __name__ == "__main__":
    main()
