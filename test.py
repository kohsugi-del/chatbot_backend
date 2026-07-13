import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(url, key)

res = supabase.rpc("vector_search", {
    "query_embedding": [0]*1536,
    "match_threshold": 0.1,
    "match_count": 3
}).execute()

print(res)