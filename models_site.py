from sqlalchemy import Column, Integer, String
from database import Base

class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    scope = Column(String, nullable=False)   # all / single / one-level
    type = Column(String, nullable=False)    # WordPress / Headless CMS / 静的HTML

    status = Column(String, nullable=False, default="pending")
