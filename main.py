from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, shutil
from dotenv import load_dotenv
from fastapi import BackgroundTasks
load_dotenv()

from rag_core import build_index, search, answer

app = FastAPI(title="RAG Chat API")

# CORS（Next.jsローカル用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    # ① DB保存
    new_site = Site(
        url=site.url,
        scope=site.scope,
        type=site.type,
    )
    db.add(new_site)
    db.commit()
    db.refresh(new_site)

    # ② バックグラウンドで ingest 実行
    background_tasks.add_task(
        ingest_site_background,
        site.url,
    )

    return new_site

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

def ingest_site_background(url: str):
    """
    サイト登録後にバックグラウンドで実行される ingest
    """
    try:
        build_index([url], [])
        print(f"[INGEST DONE] {url}")
    except Exception as e:
        print(f"[INGEST ERROR] {url}", e)

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

