import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from typing import List, Dict
from supabase_writer import save_to_supabase

print("🎉 rag_core がロードされました")

# ============ OpenAI ============
client = OpenAI()

# ============ 1. Web ============
def load_web_urls(urls: List[str]) -> List[Dict]:
    docs = []
    for url in urls:
        print(f"📘 Web取得中: {url}")
        html = requests.get(url, timeout=20).text
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        docs.append({"source": url, "text": text})
    return docs


# ============ 2. PDF ============
def load_pdfs(paths: List[str]) -> List[Dict]:
    docs = []
    for pdf in paths:
        print(f"📕 PDF読み込み中: {pdf}")
        reader = PdfReader(pdf)
        txt = ""
        for page in reader.pages:
            txt += (page.extract_text() or "") + "\n"
        docs.append({"source": pdf, "text": txt})
    return docs


# ============ 3. Chunk ============
def chunk_docs(
    documents: List[Dict],
    max_chunks: int | None = None,
) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )

    chunks = []
    for d in documents:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({
                "source": d["source"],
                "text": chunk,
            })
            if max_chunks and len(chunks) >= max_chunks:
                return chunks

    return chunks


# ============ 4. Embedding（バッチ） ============
def embed_batch(texts: List[str]):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in res.data]


# ============ 5. インデクシング ============
def build_index(
    web_urls: List[str] | None = None,
    pdf_paths: List[str] | None = None,
    max_chunks: int = 50,  # ★ 負荷制御
) -> int:

    web_urls = web_urls or []
    pdf_paths = pdf_paths or []

    docs = []
    if web_urls:
        docs.extend(load_web_urls(web_urls))
    if pdf_paths:
        docs.extend(load_pdfs(pdf_paths))

    if not docs:
        return 0

    chunks = chunk_docs(docs, max_chunks=max_chunks)
    texts = [c["text"] for c in chunks]

    print(f"🧩 chunks: {len(texts)}")

    embeddings = embed_batch(texts)

    # Supabase 保存
    for chunk, emb in zip(chunks, embeddings):
        save_to_supabase(
            content=chunk["text"],
            embedding=emb,
            source=chunk["source"],
        )

    return len(chunks)

def answer(query: str, retrieved_docs: list) -> str:
    """
    retrieved_docs: [
      ({"text": "...", "source": "..."}, similarity),
      ...
    ]
    """
    if not retrieved_docs:
        context = "該当する情報は見つかりませんでした。"
    else:
        context = "\n\n".join(
            d["text"] for d, _ in retrieved_docs
        )

    prompt = f"""
あなたは「働くあさひかわ（hataraku-asahikawa.jp）」専用の案内チャットボットです。
以下の資料だけを根拠に回答してください。
推測や一般論は書かないでください。

# 資料
{context}

# 質問
{query}

# 回答（日本語・簡潔）
"""

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return res.choices[0].message.content

