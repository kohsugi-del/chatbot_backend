# supabase_client.py
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().strip('"').strip("'").rstrip("/")
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
).strip().strip('"').strip("'")

if not SUPABASE_URL.startswith("https://"):
    raise RuntimeError(f"SUPABASE_URL is invalid: {SUPABASE_URL!r} (must start with https://)")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is not set (set SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY)")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
