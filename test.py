from supabase import create_client

url = "https://qpkgxwbrptaweahtkhjy.supabase.co"
key = "sb_secret_7YGTOkKLUdpLQSTozMOtlw_6OAEdD_w"

supabase = create_client(url, key)

res = supabase.rpc("vector_search", {
    "query_embedding": [0]*1536,
    "match_threshold": 0.1,
    "match_count": 3
}).execute()

print(res)