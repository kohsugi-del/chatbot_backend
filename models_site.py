from sqlalchemy import Column, Integer, String
from database import Base

class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    scope = Column(String, nullable=False)
    type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")

    # ★ 追加
    ingested_urls = Column(Integer, nullable=True)
