import requests
import os
import numpy as np
from dotenv import load_dotenv

# .env を読み込む
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Edge Function のエンドポイント
FN_URL = f"{SUPABASE_URL}/functions/v1/save_embedding"


def save_to_supabase(content: str, embedding: np.ndarray):
    payload = {
        "content": content,
        "embedding": embedding.tolist(),
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
    }

    try:
        res = requests.post(FN_URL, json=payload, headers=headers)
        res.raise_for_status()
        return res.json()

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("=== Supabase 保存テスト開始 ===")

    dummy_embedding = np.zeros(1536)
    response = save_to_supabase("Test content", dummy_embedding)

    print("レスポンス:", response)
