from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, shutil
from dotenv import load_dotenv
from fastapi import BackgroundTasks
from crawler_utils import resolve_urls_by_scope
from bs4 import BeautifulSoup
from models_file import File as FileModel
load_dotenv()

from rag_core import build_index, search, answer

app = FastAPI(title="RAG Chat API")

# CORS（Next.jsローカル用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(DATA_DIR, exist_ok=True)


# ==== スキーマ =====
class IngestBody(BaseModel):
    web_urls: List[str] = []
    pdf_paths: List[str] = []  # サーバー上の絶対/相対パスを渡す場合


class ChatBody(BaseModel):
    question: str
    top_k: int = 3


# ==== エンドポイント ====
@app.post("/ingest")
def ingest(body: IngestBody):
    added = build_index(body.web_urls, body.pdf_paths)
    return {"status": "ok", "added_chunks": added}


@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    # 1) 保存
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) 取り込み＆インデクシング
    added = build_index([], [save_path])
    return {"status": "ok", "path": save_path, "added_chunks": added}


@app.post("/ask")
def ask(body: ChatBody):
    retrieved = search(body.question, top_k=body.top_k)
    ans = answer(body.question, retrieved)
    refs = [{"source": d["source"], "score": float(s)} for d, s in retrieved]
    return {"answer": ans, "references": refs}

# ==== 追加：Embed 用 =====
class EmbedBody(BaseModel):
    question: str
    top_k: int = 5


@app.post("/embed")
def embed(body: EmbedBody):
    retrieved = search(body.question, top_k=body.top_k)
    ans = answer(body.question, retrieved)
    return {
        "answer": ans
    }

# ===== ここから追加 =====

from database import SessionLocal, engine
from models_site import Site
from schemas_site import SiteCreate, SiteResponse, ReingestResponse
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException

# テーブル作成
Site.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------
# POST /sites
# ----------------------------
@app.post("/sites", response_model=SiteResponse)
def create_site(
    site: SiteCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    db_site = Site(
        url=site.url,
        scope=site.scope,
        type=site.type,
        status="pending",
    )
    db.add(db_site)
    db.commit()
    db.refresh(db_site)

    # ★ これが無いと一生「準備中」
    background_tasks.add_task(
        ingest_site_background,
        db_site.id,
    )

    return db_site

# ----------------------------
# GET /sites
# ----------------------------
@app.get("/sites", response_model=list[SiteResponse])
def list_sites(db: Session = Depends(get_db)):
    return db.query(Site).order_by(Site.id.desc()).all()

# ----------------------------
# DELETE /sites/{id}
# ----------------------------
@app.delete("/sites/{site_id}")
def delete_site(site_id: int, db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    db.delete(site)
    db.commit()
    return {"status": "deleted"}

# ===== 追加ここまで =====

def ingest_site_background(site_id: int):
    print(f"[INGEST START] site_id={site_id}")  # ← 追加
    db = SessionLocal()
    try:
        site = db.query(Site).filter(Site.id == site_id).first()
        if not site:
            return

        # crawling に変更
        site.status = "crawling"
        db.commit()

        # ★ここが追加されたポイント
        urls = resolve_urls_by_scope(site.url, site.scope)

        # ★ URL数を保存
        site.ingested_urls = len(urls)
        db.commit()

        # ingest 実行
        build_index(urls, [])

        # 完了
        site.status = "done"
        db.commit()

    except Exception as e:
        site.status = "error"
        db.commit()
        print("[INGEST ERROR]", e)

    finally:
        db.close()

from fastapi import BackgroundTasks

@app.post(
        "/sites/{site_id}/reingest",
        response_model=ReingestResponse
)
def reingest_site(
    site_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    site = db.query(Site).filter(Site.id == site_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    # status を pending に戻す
    site.status = "pending"
    db.commit()

    # 再 ingest をバックグラウンド実行
    background_tasks.add_task(
        ingest_site_background,
        site.id,
    )

    return {
        "status": "reingest_started",
        "site_id": site.id,
    }

from schemas_file import FileResponse  # ★ これを追加

@app.post("/files", response_model=FileResponse)
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db_file = FileModel(
        filename=file.filename,
        path=save_path,
        status="pending",
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    background_tasks.add_task(
        ingest_file_background,
        db_file.id,
    )

    return db_file

def ingest_file_background(file_id: int):
    db = SessionLocal()
    try:
        f = db.query(FileModel).filter(FileModel.id == file_id).first()
        if not f:
            return

        f.status = "processing"
        db.commit()

        added = build_index([], [f.path])

        f.ingested_chunks = added
        f.status = "done"
        db.commit()

    except Exception as e:
        f.status = "error"
        db.commit()
        print("[FILE INGEST ERROR]", e)

    finally:
        db.close()

@app.get("/files", response_model=list[FileResponse])
def list_files(db: Session = Depends(get_db)):
    return db.query(FileModel).order_by(FileModel.id.desc()).all()

@app.delete("/files/{file_id}")
def delete_file(file_id: int, db: Session = Depends(get_db)):
    f = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not f:
        raise HTTPException(status_code=404)

    if os.path.exists(f.path):
        os.remove(f.path)

    db.delete(f)
    db.commit()
    return {"status": "deleted"}

@app.post("/files/{file_id}/reingest")
def reingest_file(
    file_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    f = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not f:
        raise HTTPException(status_code=404)

    f.status = "pending"
    db.commit()

    background_tasks.add_task(
        ingest_file_background,
        f.id,
    )

    return {"status": "reingest_started"}