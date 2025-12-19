from sqlalchemy import Column, Integer, String
from database import Base

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    ingested_chunks = Column(Integer, nullable=True)
