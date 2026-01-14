# ingest_dev.py
from database import SessionLocal
from models_site import Site
from models_file import File as FileModel
from rag_core import build_index

def run():
    db = SessionLocal()

    # pending の site を1件だけ拾う
    site = db.query(Site).filter(Site.status == "pending").first()
    if site:
        print(f"[DEV INGEST] site_id={site.id}")
        site.status = "processing"
        db.commit()

        # URL収集（仮で1件だけ）
        urls = [site.url]
        added = build_index(urls, max_chunks=20)

        site.status = "done"
        site.ingested_chunks = added
        db.commit()

    else:
        print("No pending sites")

    db.close()

if __name__ == "__main__":
    run()
