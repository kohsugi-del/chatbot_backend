import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

REST_URL = f"{SUPABASE_URL}/rest/v1/vectors"

client = OpenAI()

def save_vector(content: str):
    # 1. OpenAI 埋め込み
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=content
    ).data[0].embedding

    # 2. Supabase へ保存（REST API）
    headers = {
        "apikey": SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    data = {
        "content": content,
        "embedding": emb
    }

    res = requests.post(REST_URL, json=data, headers=headers)

    print("Status:", res.status_code)
    print("Response:", res.text)

# テスト
save_vector("これは Supabase REST API で保存したテスト文章です。")
