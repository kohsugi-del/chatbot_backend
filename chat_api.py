from fastapi import FastAPI
from pydantic import BaseModel
import requests
from openai import OpenAI

SUPABASE_URL = "https://qpkgxwbrptaweahtkhjy.supabase.co"
SERVICE_ROLE_KEY = "sb_secret_7YGTOkKLUdpLQSTozMOtlw_6OAEdD_w"

client = OpenAI()

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):

    # 1) 質問の embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.question
    ).data[0].embedding

    # 2) Supabase match_chunks 検索
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_chunks"

    payload = {
        "query_embedding": embedding,
        "match_count": 5
    }

    headers = {
        "Content-Type": "application/json",
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}"
    }

    res = requests.post(url, json=payload, headers=headers)
    matches = res.json()

    # 3) RAG の system プロンプトを作成
    context_text = "\n".join([m["content"] for m in matches])

    prompt = f"""
あなたは親切な回答AIです。
以下の資料に基づいて、質問に正確に答えてください。

【資料】
{context_text}

【質問】
{req.question}

必ず資料に基づいて答えてください。
"""

    # 4) GPT で最終回答
    answer = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    ).choices[0].message["content"]

    return {
        "answer": answer,
        "matches": matches
    }
