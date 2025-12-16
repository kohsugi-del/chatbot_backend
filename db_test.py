import os
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

url = os.environ.get("DATABASE_URL")
print("DATABASE_URL:", url)

try:
    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("select now() as time;")
            print("Connected OK:", cur.fetchone())
except Exception as e:
    print("Connection ERROR:", e)
