from datetime import datetime
from typing import Optional

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from database import (
    User,
    Conversation,
    Message,
    get_db,
    init_db,
)

from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)


app = FastAPI(
    title="AI Chat API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


init_db()


# =========================
# MODELS
# =========================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ConversationRequest(BaseModel):
    title: Optional[str] = "Cuộc trò chuyện mới"
    tab: Optional[str] = "chat"


class MessageRequest(BaseModel):
    content: str


# =========================
# ROOT
# =========================

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "AI Chat API đang hoạt động"
    }


# =========================
# REGISTER
# =========================

@app.post("/api/auth/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db),
):

    email = data.email.lower().strip()

    if len(data.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu phải có ít nhất 6 ký tự."
        )

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email đã được đăng ký."
        )

    user = User(
        email=email,
        password_hash=hash_password(data.password),
        settings_json="{}",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Đăng ký xong = đăng nhập luôn
    token = create_access_token(user.id)

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
        },
    }


# =========================
# LOGIN
# =========================

@app.post("/api/auth/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db),
):

    email = data.email.lower().strip()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Email hoặc mật khẩu không đúng."
        )

    if not user.password_hash:
        raise HTTPException(
            status_code=400,
            detail="Tài khoản này không đăng nhập bằng mật khẩu."
        )

    if not verify_password(
        data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Email hoặc mật khẩu không đúng."
        )

    token = create_access_token(user.id)

    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
        },
    }


# =========================
# KIỂM TRA ĐĂNG NHẬP
# =========================

@app.get("/api/auth/me")
def me(
    current_user: User = Depends(get_current_user)
):

    return {
        "id": current_user.id,
        "email": current_user.email,
        "created_at": current_user.created_at,
    }


# =========================
# TẠO CUỘC TRÒ CHUYỆN
# =========================

@app.post("/api/conversations")
def create_conversation(
    data: ConversationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    conversation = Conversation(
        user_id=current_user.id,
        title=data.title or "Cuộc trò chuyện mới",
        tab=data.tab or "chat",
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "id": conversation.id,
        "title": conversation.title,
        "tab": conversation.tab,
    }


# =========================
# LẤY DANH SÁCH LỊCH SỬ
# =========================

@app.get("/api/conversations")
def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .all()
    )

    return [
        {
            "id": conversation.id,
            "title": conversation.title,
            "tab": conversation.tab,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
        }
        for conversation in conversations
    ]


# =========================
# LẤY MỘT CUỘC TRÒ CHUYỆN
# =========================

@app.get("/api/conversations/{conversation_id}")
def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Không tìm thấy cuộc trò chuyện."
        )

    return {
        "id": conversation.id,
        "title": conversation.title,
        "tab": conversation.tab,
        "messages": [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
            }
            for message in conversation.messages
        ],
    }


# =========================
# LƯU TIN NHẮN
# =========================

@app.post(
    "/api/conversations/{conversation_id}/messages"
)
def save_message(
    conversation_id: int,
    data: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Cuộc trò chuyện không tồn tại."
        )

    message = Message(
        conversation_id=conversation.id,
        role="user",
        content=data.content,
    )

    db.add(message)

    conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    return {
        "success": True,
        "message": {
            "id": message.id,
            "role": message.role,
            "content": message.content,
        },
    }


# =========================
# LƯU CÂU TRẢ LỜI AI
# =========================

@app.post(
    "/api/conversations/{conversation_id}/assistant"
)
def save_assistant_message(
    conversation_id: int,
    data: MessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Cuộc trò chuyện không tồn tại."
        )

    message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=data.content,
    )

    db.add(message)

    conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    return {
        "success": True,
        "message": {
            "id": message.id,
            "role": message.role,
            "content": message.content,
        },
    }


# =========================
# XÓA CUỘC TRÒ CHUYỆN
# =========================

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Cuộc trò chuyện không tồn tại."
        )

    db.delete(conversation)
    db.commit()

    return {
        "success": True,
        "message": "Đã xóa cuộc trò chuyện."
    }


# =========================
# NHẬN DIỆN YÊU CẦU TẠO ẢNH
# =========================

IMAGE_KEYWORDS = [
    "tạo ảnh",
    "tạo hình ảnh",
    "vẽ ảnh",
    "tạo hình",
    "generate image",
    "create image",
    "create an image",
    "generate a picture",
    "make an image",
    "draw an image",
]


def is_image_request(text: str) -> bool:
    text = text.lower().strip()

    return any(
        keyword in text
        for keyword in IMAGE_KEYWORDS
    )


@app.post("/api/detect-request")
def detect_request(
    data: MessageRequest,
):

    return {
        "is_image_request": is_image_request(
            data.content
        ),
        "content": data.content,
    }
