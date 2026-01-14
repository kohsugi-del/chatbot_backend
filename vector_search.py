from supabase_client import supabase
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed_query(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding


def search(query: str, top_k: int = 3):
    """
    Supabase Vector Search を使った検索
    """
    q_emb = embed_query(query)

    # Supabase RPC（後で作る関数）を呼ぶ想定
    res = supabase.rpc(
        "match_documents",
        {
            "query_embedding": q_emb,
            "match_count": top_k,
        },
    ).execute()

    results = []
    for row in res.data:
        results.append((
            {
                "text": row["content"],
                "source": row.get("source", ""),
            },
            row["similarity"],
        ))

    return results
