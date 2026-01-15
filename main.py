from fastapi import FastAPI, UploadFile, File as FastAPIFile, Depends, HTTPException, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, shutil, re, time, uuid
from dotenv import load_dotenv

load_dotenv()

from rag_core import answer
from vector_search import search

# =========================
# App
# =========================
app = FastAPI()

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

# --- preflight保険（残してOK） ---
@app.options("/{rest_of_path:path}")
def preflight_handler(rest_of_path: str, request: Request):
    return Response(status_code=200)

@app.get("/__ping")
def ping():
    return {"ok": True}


# =========================
# CORS
# =========================
# NOTE:
# ここは「app作成直後」に置くのが正解
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

# -------------------------
# (重要) プリフライト(OPTIONS)を確実に200で返す保険
# - 環境/ルータ/例外状況によってはミドルウェアより先に落ちるケースがあるため
# - これで "No 'Access-Control-Allow-Origin' header" を潰しやすい
# -------------------------
@app.options("/{rest_of_path:path}")
def preflight_handler(rest_of_path: str, request: Request):
    # CORSミドルウェアがヘッダを付けるので、ここは空でOK
    return Response(status_code=200)

# -------------------------
# ヘルスチェック（CORS確認用）
# -------------------------
@app.get("/__ping")
def ping():
    return {"ok": True}

# =========================
# Chat API
# =========================
class ChatBody(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None
    top_k: int = 8

def get_question(body: ChatBody) -> str:
    q = (body.question or body.message or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question (or message) is required")
    return q

def build_refs(retrieved):
    refs = []
    for d, s in retrieved:
        try:
            src = d.get("source", "") if isinstance(d, dict) else ""
        except Exception:
            src = ""
        refs.append({"source": src, "score": float(s)})
    return refs

@app.post("/chat")
def chat(body: ChatBody):
    q = get_question(body)
    retrieved = search(q, top_k=body.top_k)
    ans = answer(q, retrieved)
    refs = build_refs(retrieved)
    return {"answer": ans, "references": refs}

@app.post("/ask")
def ask(body: ChatBody):
    q = get_question(body)
    retrieved = search(q, top_k=body.top_k)
    ans = answer(q, retrieved)
    refs = build_refs(retrieved)
    return {"answer": ans, "references": refs}

@app.post("/embed")
def embed(body: ChatBody):
    q = get_question(body)
    retrieved = search(q, top_k=body.top_k)
    ans = answer(q, retrieved)
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
        status="pending",
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

# Windows / ブラウザの filename を安全化
_invalid_chars = re.compile(r'[<>:"/\\|?*\x00-\x1F]')  # Windows禁止 + 制御文字

def safe_filename(original: str) -> str:
    # ブラウザによって fakepath が混ざる場合があるので basename 化
    name = os.path.basename(original or "")
    name = name.strip()
    if not name:
        name = "upload.pdf"
    name = _invalid_chars.sub("_", name)
    # 末尾のドット・スペースはWindowsで危険
    name = name.rstrip(". ").strip()
    if not name:
        name = "upload.pdf"
    return name

def unique_path(dir_path: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    # 衝突回避（uuid）
    uid = uuid.uuid4().hex[:8]
    return os.path.join(dir_path, f"{base}_{uid}{ext}")

@app.post("/files", response_model=FileResponse)
def upload_file(
    file: UploadFile = FastAPIFile(...),
    db: Session = Depends(get_db),
):
    try:
        fn = safe_filename(file.filename)
        save_path = unique_path(DATA_DIR, fn)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # ✅ B案：DBには filename だけ
        db_file = FileModel(
            filename=os.path.basename(save_path),
            error_message=None,
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        return db_file

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload failed: {type(e).__name__}: {e}")
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        return db_file

    except Exception as e:
        # 接続を切らずに必ずHTTPで返す
        raise HTTPException(status_code=500, detail=f"upload failed: {type(e).__name__}: {e}")


@app.get("/files")
def list_files(db: Session = Depends(get_db)):
    rows = db.query(FileModel).order_by(FileModel.id.desc()).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "status": "uploaded",          # ← 追加（UI用）
            "ingested_chunks": 0,          # ← 追加（UI用）
            "error_message": r.error_message,
        }
        for r in rows
    ]


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

    try:
        file_path = os.path.join(DATA_DIR, f.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"remove failed: {type(e).__name__}: {e}")

    db.delete(f)
    db.commit()
    return {"status": "deleted"}


# =========================
# ingest 実行（バックグラウンド）
# =========================
from ingest import ingest_site_from_db

@app.post("/sites/{site_id}/reingest_local")
def reingest_local(site_id: int, background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_site_from_db, site_id)
    return {"status": "queued", "site_id": site_id}


from pdf_ingest import ingest_pdf

@app.post("/files/{file_id}/ingest_local")
def ingest_file_local(file_id: int, db: Session = Depends(get_db)):
    f = db.query(FileModel).filter(FileModel.id == file_id).first()
    if not f:
        raise HTTPException(status_code=404, detail="File not found")

    pdf_path = os.path.join(DATA_DIR, f.filename)

    try:
        from pdf_ingest import ingest_pdf  # あとで作るファイル
        result = ingest_pdf(file_id=file_id, pdf_path=pdf_path, file_name=f.filename)
        return {"status": "done", "file_id": file_id, **result}

    except Exception as e:
        # エラーを files テーブルの error_message に残す
        try:
            f.error_message = f"{type(e).__name__}: {e}"
            db.commit()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"ingest failed: {type(e).__name__}: {e}")
