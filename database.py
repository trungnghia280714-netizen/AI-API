import os
from datetime import datetime

from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")

# Render cấp URL Postgres dạng "postgres://" nhưng SQLAlchemy cần "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=True)  # NULL nếu tài khoản đăng nhập bằng Google
    created_at = Column(DateTime, default=datetime.utcnow)
    settings_json = Column(Text, default="{}")  # lưu cài đặt (theme, v.v.) dạng JSON string
    plan = Column(String, default="free")  # "free" / "inteligent_cold" / "inteligent_super_cold"

    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tab = Column(String, default="chat")  # "chat" hoặc "code"
    title = Column(String, default="Cuộc trò chuyện mới")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" hoặc "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class UsageLog(Base):
    """Đếm số lần dùng mỗi tính năng trong 1 ngày, theo từng user -> áp dụng hạn mức gói free."""
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    feature = Column(String, nullable=False)  # "chat" / "code" / "image" / "video"
    usage_date = Column(String, nullable=False)  # "YYYY-MM-DD" (giờ UTC)
    count = Column(Integer, default=0)


def init_db():
    Base.metadata.create_all(bind=engine)
    # create_all() chỉ tạo BẢNG MỚI, không tự thêm CỘT MỚI vào bảng đã tồn tại.
    # Tự vá các cột có thể bị thiếu do nâng cấp code sau khi bảng đã được tạo từ trước.
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan VARCHAR DEFAULT 'free'"))
    except Exception:
        pass  # bỏ qua nếu DB không hỗ trợ cú pháp này (vd: SQLite) hoặc cột đã tồn tại


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
