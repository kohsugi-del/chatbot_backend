# database.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# もし postgresql:// のまま来ても psycopg に寄せる
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]

# もし間違って psycopg2 になってたら psycopg に戻す
DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
