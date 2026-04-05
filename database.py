# database.py
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set (Render env var or .env locally).")

# ★ sqlite の相対パスを「このファイル(database.py)の場所基準」で固定する
# .env は DATABASE_URL=sqlite:///./app.db のままでOK
if DATABASE_URL.startswith("sqlite:///./"):
    filename = DATABASE_URL.replace("sqlite:///./", "")
    db_path = (Path(__file__).resolve().parent / filename)
    DATABASE_URL = f"sqlite:///{db_path.as_posix()}"

# postgresql:// でも psycopg に寄せる
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]

# psycopg2 指定が来ても psycopg に戻す
DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+psycopg://")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
