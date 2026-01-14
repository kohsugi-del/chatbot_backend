import requests
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

FN_URL = f"{SUPABASE_URL}/functions/v1/save_embedding"


def save_to_supabase(content: str, embedding, source: str | None = None):
    payload = {
        "content": content,
        "embedding": embedding if isinstance(embedding, list) else embedding.tolist(),
    }

    if source:
        payload["source"] = source

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
    }

    res = requests.post(FN_URL, json=payload, headers=headers)
    res.raise_for_status()
    return res.json()

