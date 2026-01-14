from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, shutil
from dotenv import load_dotenv

load_dotenv()

from rag_core import answer
from vector_search import search

# =========================
# App
# =========================
app = FastAPI(title="RAG Chat API (Lightweight)")

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

# =========================
# Chat API（軽い）
# =========================
class ChatBody(BaseModel):
    question: str
    top_k: int = 3


@app.post("/ask")
def ask(body: ChatBody):
    retrieved = search(body.question, top_k=body.top_k)
    ans = answer(body.question, retrieved)
    refs = [{"source": d["source"], "score": float(s)} for d, s in retrieved]
    return {"answer": ans, "references": refs}


@app.post("/embed")
def embed(body: ChatBody):
    retrieved = search(body.question, top_k=body.top_k)
    ans = answer(body.question, retrieved)
    return {"answer": ans}

# =========================
# DB（Site / File 管理のみ）
# =========================
from database import SessionLocal, engine
from sqlalchemy.orm import Session
from models_site import Site
from models_file import File as FileModel
from schemas_site import SiteCreate, SiteResponse, ReingestResponse
from schemas_file import FileResponse

Site.metadata.create_all(bind=engine)
FileModel.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# Sites
# -------------------------
@app.post("/sites", response_model=SiteResponse)
def create_site(site: SiteCreate, db: Session = Depends(get_db)):
    db_site = Site(
        url=site.url,
        scope=site.scope,
        type=site.type,
        status="pending",   # ← ingest は外部がやる
    )
    db.add(db_site)
    db.commit()
    db.refresh(db_site)
    return db_site


@app.get("/sites", response_model=List[SiteResponse])
def list_sites(db: Session = Depends(get_db)):
    return db.query(Site).order_by(Site.id.desc()).all()


@app.post("/sites/{site_id}/reingest", response_model=ReingestResponse)
def reingest_site(site_id: int, db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    site.status = "pending"
    db.commit()

    # ★ ingest は GitHub Actions が拾う
    return {"status": "queued", "site_id": site.id}


@app.delete("/sites/{site_id}")
def delete_site(site_id: int, db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id).first()
    if not site:
        raise HTTPException(status_code=404)

    db.delete(site)
    db.commit()
    return {"status": "deleted"}

# -------------------------
# Files
# -------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(DATA_DIR, exist_ok=True)


@app.post("/files", response_model=FileResponse)
def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db_file = FileModel(
        filename=file.filename,
        path=save_path,
        status="pending",   # ← ingest は外部
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file


@app.get("/files", response_model=List[FileResponse])
def list_files(db: Session = Depends(get_db)):
    return db.query(FileModel).order_by(FileModel.id.desc()).all()


@app.post("/files/{file_id}/reingest")
def reingest_file(file_id: int, db: Session = Depends(get_db)):
    f = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not f:
        raise HTTPException(status_code=404)

    f.status = "pending"
    db.commit()

    return {"status": "queued", "file_id": f.id}


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
