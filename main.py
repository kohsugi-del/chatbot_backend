from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, shutil
from dotenv import load_dotenv
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


