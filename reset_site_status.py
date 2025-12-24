import sqlite3

DB_PATH = "sites.db"  # ★ database.py と一致させる

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 現在の状態確認
cur.execute("SELECT id, url, status FROM sites;")
rows = cur.fetchall()
print("Before:", rows)

# status を pending に戻す
cur.execute(
    "UPDATE sites SET status = 'pending' WHERE id = 1;"
)
conn.commit()

# 更新後確認
cur.execute("SELECT id, url, status FROM sites;")
rows = cur.fetchall()
print("After:", rows)

conn.close()
