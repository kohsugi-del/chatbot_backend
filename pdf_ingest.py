# pdf_ingest.py
import os
import uuid
from typing import List
from dotenv import load_dotenv
load_dotenv()

import pypdf
from openai import OpenAI
from supabase import create_client

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # ローカルなら service role 推奨
)

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def embed_texts(chunks: List[str]) -> List[List[float]]:
    res = client.embeddings.create(model=EMBED_MODEL, input=chunks)
    return [d.embedding for d in res.data]

def ingest_pdf(file_id: int, pdf_path: str, file_name: str):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise RuntimeError("PDFからテキストが抽出できませんでした（スキャンPDFの可能性）")

    chunks = chunk_text(text)
    vectors = embed_texts(chunks)

    # ✅ rag_chunks の列に合わせる：id / content / embedding のみ
    rows = []
    for c, v in zip(chunks, vectors):
        rows.append({
            "id": str(uuid.uuid4()),  # uuid
            "content": c,
            "embedding": v,
        })

    supabase.table("rag_chunks").insert(rows).execute()

    return {"ingested_chunks": len(chunks)}
