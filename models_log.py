import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from database import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class SessionLog(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    municipality_id = Column(String, nullable=False, default="htrk-asahikawa")
    started_at = Column(DateTime(timezone=True), default=_now)


class TurnLog(Base):
    __tablename__ = "turns"

    id = Column(String(36), primary_key=True, default=_uuid)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=True)
    turn_order = Column(Integer, nullable=False)
    role = Column(String, nullable=False)       # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    user_type = Column(Text, nullable=True)     # 'jobseeker' / 'company' / 'other'
    topic_type = Column(Text, nullable=True)    # 'job' / 'site_usage' / 'other'
    bias_type = Column(Text, nullable=True)     # 'loss_aversion' / 'status_quo' / 'choice_overload'
    created_at = Column(DateTime(timezone=True), default=_now)
