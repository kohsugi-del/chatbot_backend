import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import numpy as np
from typing import List, Dict
from supabase_writer import save_to_supabase

print("🎉 rag_core がロードされました")

# ============ 0. OpenAIクライアント ============
client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用

# ======== グローバル（メモリ内DB） =========
CHUNKS: List[Dict] = []          # {"source": str, "text": str}
EMBEDDINGS: List[np.ndarray] = []


# ============ 1. スクレイピング（Webページ） ============
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


# ============ 2. PDF取り込み ============
def load_pdfs(paths: List[str]) -> List[Dict]:
    docs = []
    for pdf in paths:
        print(f"📕 PDF読み込み中: {pdf}")
        reader = PdfReader(pdf)
        txt = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            txt += page_text + "\n"
        docs.append({"source": pdf, "text": txt})
    return docs


# ============ 3. チャンク化 ============
def chunk_docs(documents: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = []
    for d in documents:
        for chunk in splitter.split_text(d["text"]):
            chunks.append({"source": d["source"], "text": chunk})
    return chunks


# ============ 4. OpenAI 埋め込み ============
def embed(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)


# ============ 5. 類似検索 ============
def search(query: str, top_k: int = 3):
    if not CHUNKS or not EMBEDDINGS:
        return []
    q_emb = embed(query)
    scores = []
    qn = np.linalg.norm(q_emb)
    for i, emb in enumerate(EMBEDDINGS):
        score = float(np.dot(q_emb, emb) / (qn * np.linalg.norm(emb)))
        scores.append((score, i))
    scores.sort(reverse=True)
    top = scores[:top_k]
    return [ (CHUNKS[i], s) for s, i in top ]


# ============ 6. LLM で回答生成（RAG） ============
def answer(query: str, retrieved_docs: List):
    context = "\n\n".join([d["text"] for d, _ in retrieved_docs])
    prompt = f"""
あなたは「働くあさひかわ（hataraku-asahikawa.jp）」専用の案内チャットボットです。
以下の資料とウェブサイト情報だけを根拠に回答してください。

# 資料
{context}

# 質問
{query}

# 回答（端的に）
"""
    res = client.chat.completions.create(
        model="gpt-4.1-mini",  # ←あなたの指定どおり
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


# ============ 7. インデクシング（公開関数） ============
def build_index(web_urls: List[str], pdf_paths: List[str]) -> int:
    global CHUNKS, EMBEDDINGS

    web_docs = load_web_urls(web_urls) if web_urls else []
    pdf_docs = load_pdfs(pdf_paths) if pdf_paths else []
    all_docs = web_docs + pdf_docs

    new_chunks = chunk_docs(all_docs)
    new_embeddings = [embed(c["text"]) for c in new_chunks]

    CHUNKS.extend(new_chunks)
    EMBEDDINGS.extend(new_embeddings)

    # ★ Supabaseへ保存
    for chunk, emb in zip(new_chunks, new_embeddings):
        res = save_to_supabase(chunk["text"], emb)
        print("Saved:", res)

    return len(new_chunks)