# save_real_vector.py
import numpy as np
import requests
from openai import OpenAI

SUPABASE_URL = "https://qpkgxwbrptaweahtkhjy.supabase.co"
SERVICE_ROLE_KEY = "sb_secret_7YGTOkKLUdpLQSTozMOtlw_6OAEdD_w"

FN_URL = f"{SUPABASE_URL}/functions/v1/save_embedding"

client = OpenAI()

def save_text(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    payload = {
        "content": text,
        "embedding": emb,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
        "apikey": SERVICE_ROLE_KEY,
    }

    res = requests.post(FN_URL, json=payload, headers=headers)
    print("Status:", res.status_code)
    print("Response:", res.text)


if __name__ == "__main__":
    save_text("旭川市で仕事を探すための情報サイトです。")
