# backend/search_vector.py
import os
import numpy as np
import requests
from openai import OpenAI

SUPABASE_URL = "https://qpkgxwbrptaweahtkhjy.supabase.co"  # ←置き換え
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # .env 等で管理推奨

def vector_search(query_emb: np.ndarray, match_count: int = 5):
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_rag_chunks"

    payload = {
        "query_embedding": query_emb.tolist(),
        "match_count": match_count,
    }

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,                      # 小文字が重要
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",   # Bearer も必要
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    res = requests.post(url, json=payload, headers=headers, timeout=30)
    res.raise_for_status()
    return res.json()

if __name__ == "__main__":
    # 1) OpenAI で埋め込み作成
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input="働くあさひかわについて教えてください"
    ).data[0].embedding

    # 2) Supabase RPC を叩く
    results = vector_search(np.array(emb), match_count=5)

    print("Status: OK")
    for r in results:
        print(f"- id={r['id']} sim={r['similarity']:.4f} content={r['content'][:40]}...")
