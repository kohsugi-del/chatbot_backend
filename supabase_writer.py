import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
).strip()

_client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_to_supabase(
    content: str,
    embedding,
    source: str | None = None,
    title: str | None = None,
) -> dict:
    """documentsテーブルに1チャンクをINSERT（Edgeファンクション不要）"""
    emb = embedding if isinstance(embedding, list) else list(embedding)

    row: dict = {"content": content, "embedding": emb}
    if source:
        row["source"] = source
        row["source_url"] = source
    if title:
        row["title"] = title

    result = _client.table("documents").insert(row).execute()
    return result.data[0] if result.data else {}
