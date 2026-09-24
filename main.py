import json
import os
import re
from datetime import date

import requests
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy.orm import Session

from auth import (
    create_access_token,
    get_current_user,
    get_optional_user,
    hash_password,
    verify_password,
)
from database import Conversation, Message, UsageLog, User, get_db, init_db

# =====================================================================
# Cấu hình từ biến môi trường (KHÔNG hardcode key)
# =====================================================================
def _parse_keys(env_name: str) -> list:
    """Mỗi biến có thể chứa NHIỀU key cách nhau bằng dấu phẩy.
    Khi 1 key bị giới hạn (429) hoặc lỗi xác thực, tự động thử key kế tiếp."""
    raw = os.environ.get(env_name, "")
    return [k.strip() for k in raw.split(",") if k.strip()]

# --- API Keys ---
APINEX_API_KEYS      = _parse_keys("APINEX_API_KEY")
CODECRAFT_API_KEYS   = _parse_keys("CODECRAFT_API_KEY")   # Claude Sonnet 5
OPENROUTER_TTS_KEY   = os.environ.get("OPENROUTER_TTS_KEY", "")  # Fish Audio TTS

# --- Base URLs ---
APINEX_BASE_URL    = os.environ.get("APINEX_BASE_URL",    "https://apinex.bond/v1")
CODECRAFT_BASE_URL = os.environ.get("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1")

# =====================================================================
# Danh sách model chat
# Lưu ý: Apinex yêu cầu tiền tố "free/" cho các model miễn phí
# Weight của Apinex: ×4 = Gemini Flash, ×3 = GPT Luna, ×2 = các model còn lại
# Pool chung 1 triệu token/ngày reset 00:00 UTC — dùng model weight thấp để tiết kiệm quota
# =====================================================================
MODEL_CATALOG = {
    "deepseek-v4-1-flash": {
        "label": "DeepSeek V4.1 Flash",
        "url": f"{APINEX_BASE_URL}/chat/completions",
        "keys": APINEX_API_KEYS,
        "model": "free/deepseek-v4.1-flash",   # weight ×5 — tiết kiệm nhất
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "url": f"{APINEX_BASE_URL}/chat/completions",
        "keys": APINEX_API_KEYS,
        "model": "free/deepseek-v4-pro-0813",   # weight ×5
    },
    "gemini-3-8-flash": {
        "label": "Gemini 3.8 Flash",
        "url": f"{APINEX_BASE_URL}/chat/completions",
        "keys": APINEX_API_KEYS,
        "model": "free/gemini-3.8-flash",       # weight ×5
    },
    "gemini-3-1-pro": {
        "label": "Gemini 3.1 Pro",
        "url": f"{APINEX_BASE_URL}/chat/completions",
        "keys": APINEX_API_KEYS,
        "model": "free/gemini-3.1-pro",         # weight ×
    },
    "gpt-5-6-luna": {
        "label": "GPT 6 Luna",
        "url": f"{APINEX_BASE_URL}/chat/completions",
        "keys": APINEX_API_KEYS,
        "model": "free/gpt-6-luna",           # weight ×3
    },
    "claude-sonnet-5": {
        "label": "Claude Opus 5",
        "url": f"{CODECRAFT_BASE_URL}/chat/completions",
        "keys": CODECRAFT_API_KEYS,
        "model": "claude-opus-5",
    },
}
DEFAULT_MODEL_ID = "deepseek-v4-1-flash"   # weight ×2 — mặc định tiết kiệm quota nhất

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Hạn mức gói free / ngày (theo tài khoản đã đăng nhập)
DAILY_LIMITS = {"chat": 40}
FEATURE_NAMES_VI = {"chat": "Chat"}
UNLIMITED_PLANS = {"inteligent_cold", "inteligent_super_cold"}


def check_and_increment_usage(db: Session, user, feature: str):
    """Trả về None nếu còn hạn mức (đã tăng đếm),
    hoặc chuỗi lỗi nếu đã hết hạn mức hôm nay.
    Nếu user=None (khách chưa đăng nhập) hoặc user có gói không giới hạn -> bỏ qua."""
    if not user:
        return None
    if getattr(user, "plan", "free") in UNLIMITED_PLANS:
        return None

    limit = DAILY_LIMITS.get(feature)
    if not limit:
        return None

    today = date.today().isoformat()
    log = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id, UsageLog.feature == feature, UsageLog.usage_date == today)
        .first()
    )

    if log and log.count >= limit:
        name = FEATURE_NAMES_VI.get(feature, feature)
        return f"Bạn đã dùng hết {limit} lượt {name} miễn phí hôm nay. Vui lòng quay lại vào ngày mai."

    if log:
        log.count += 1
    else:
        log = UsageLog(user_id=user.id, feature=feature, usage_date=today, count=1)
        db.add(log)
    db.commit()
    return None


app = FastAPI(title="INTELIGENT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# =====================================================================
# Gọi model chat theo model_id trong MODEL_CATALOG
# =====================================================================
def call_chat_model(model_id: str, messages: list, temperature: float = 0.7, max_tokens: int = 4096):
    entry = MODEL_CATALOG.get(model_id)
    if not entry:
        raise ValueError(f"Model '{model_id}' không tồn tại.")
    if not entry["keys"]:
        raise ValueError(f"Server chưa cấu hình key cho model '{entry['label']}'.")

    body = {
        "model": entry["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Một số model cần tham số extra (vd: DeepSeek reasoning trên NVIDIA)
    if "extra_body" in entry:
        body.update(entry["extra_body"])

    last_error = None
    for key in entry["keys"]:
        try:
            resp = requests.post(
                entry["url"],
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            # DeepSeek reasoning trả phần suy luận riêng — lấy content câu trả lời cuối
            if not content:
                reasoning = message.get("reasoning") or message.get("reasoning_content")
                if reasoning:
                    content = "(Model đang suy luận nhưng chưa kịp trả lời — thử tăng max_tokens)"
            return content
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code in (401, 429):
                continue  # key này hết hạn mức hoặc sai -> thử key kế tiếp
            raise
    raise last_error


def extract_openai_style_error(e: requests.exceptions.HTTPError) -> str:
    try:
        return e.response.json().get("error", {}).get("message", "")
    except Exception:
        return ""


def save_turn(db: Session, user: User, conversation_id, tab: str, user_text: str, assistant_text: str):
    """Lưu 1 lượt hỏi-đáp vào DB nếu người dùng đã đăng nhập. Trả về conversation_id."""
    conv = None
    if conversation_id:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
            .first()
        )
    if not conv:
        title = user_text.strip()[:60] or "Cuộc trò chuyện mới"
        conv = Conversation(user_id=user.id, tab=tab, title=title)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    db.add(Message(conversation_id=conv.id, role="user", content=user_text))
    db.add(Message(conversation_id=conv.id, role="assistant", content=assistant_text))
    db.commit()
    return conv.id


# =====================================================================
# AUTH
# =====================================================================
@app.post("/api/auth/register")
async def register(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not EMAIL_RE.match(email):
        return JSONResponse({"error": "Email không hợp lệ."}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"error": "Mật khẩu phải có ít nhất 6 ký tự."}, status_code=400)

    if db.query(User).filter(User.email == email).first():
        return JSONResponse({"error": "Email này đã được đăng ký."}, status_code=409)

    user = User(email=email, password_hash=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return {"token": token, "email": user.email}


@app.post("/api/auth/login")
async def login(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return JSONResponse({"error": "Email hoặc mật khẩu không đúng."}, status_code=401)

    token = create_access_token(user.id)
    return {"token": token, "email": user.email}


@app.post("/api/auth/google")
async def google_login(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    credential = body.get("credential", "")
    if not credential:
        return JSONResponse({"error": "Thiếu 'credential' (Google ID token)."}, status_code=400)

    if not GOOGLE_CLIENT_ID:
        return JSONResponse({"error": "Server chưa cấu hình GOOGLE_CLIENT_ID."}, status_code=400)

    try:
        payload = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        email = payload.get("email", "").strip().lower()
        if not email:
            return JSONResponse({"error": "Không lấy được email từ tài khoản Google."}, status_code=400)

        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(email=email, password_hash=None)
            db.add(user)
            db.commit()
            db.refresh(user)

        token = create_access_token(user.id)
        return {"token": token, "email": user.email}

    except ValueError as e:
        return JSONResponse({"error": f"Xác thực Google thất bại: {str(e)}"}, status_code=401)
    except Exception as e:
        return JSONResponse({"error": f"Lỗi máy chủ khi đăng nhập Google: {str(e)}"}, status_code=500)


@app.get("/api/auth/me")
async def me(user: User = Depends(get_current_user)):
    return {
        "email": user.email,
        "plan": user.plan or "free",
        "settings": json.loads(user.settings_json or "{}"),
    }


# =====================================================================
# MODELS (danh sách model cho ô chọn trên giao diện)
# =====================================================================
@app.get("/api/models")
async def list_models():
    return [
        {"id": model_id, "label": entry["label"], "available": bool(entry["keys"])}
        for model_id, entry in MODEL_CATALOG.items()
    ]


# =====================================================================
# CONVERSATIONS (lịch sử trò chuyện)
# =====================================================================
@app.get("/api/conversations")
async def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
        .all()
    )
    return [
        {"id": c.id, "title": c.title, "tab": c.tab, "updated_at": c.updated_at.isoformat()}
        for c in convs
    ]


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        return JSONResponse({"error": "Không tìm thấy cuộc trò chuyện."}, status_code=404)

    return {
        "id": conv.id,
        "title": conv.title,
        "tab": conv.tab,
        "messages": [{"role": m.role, "content": m.content} for m in conv.messages],
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user.id)
        .first()
    )
    if not conv:
        return JSONResponse({"error": "Không tìm thấy cuộc trò chuyện."}, status_code=404)
    db.delete(conv)
    db.commit()
    return {"ok": True}


# =====================================================================
# SETTINGS
# =====================================================================
@app.get("/api/settings")
async def get_settings(user: User = Depends(get_current_user)):
    return json.loads(user.settings_json or "{}")


@app.put("/api/settings")
async def update_settings(
    request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    current = json.loads(user.settings_json or "{}")
    current.update(body)
    user.settings_json = json.dumps(current)
    db.add(user)
    db.commit()
    return current


# =====================================================================
# CHAT (có chọn model)
# =====================================================================
@app.post("/api/chat")
async def chat(
    request: Request,
    user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    message = body.get("message", "").strip()
    history = body.get("history", [])
    conversation_id = body.get("conversation_id")
    model_id = body.get("model_id", DEFAULT_MODEL_ID)

    if not message:
        return JSONResponse({"error": "Thiếu 'message'."}, status_code=400)

    if model_id not in MODEL_CATALOG:
        return JSONResponse({"error": f"Model '{model_id}' không hợp lệ."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "chat")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    messages = history + [{"role": "user", "content": message}]

    try:
        reply = call_chat_model(model_id, messages)
        result = {"reply": reply}
        if user:
            result["conversation_id"] = save_turn(db, user, conversation_id, "chat", message, reply)
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = extract_openai_style_error(e)
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {str(e)}"}, status_code=502)


# =====================================================================
# TTS — Text to Speech qua Fish Audio (OpenRouter)
# =====================================================================
@app.post("/api/tts")
async def tts(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Thiếu 'text'."}, status_code=400)

    if not OPENROUTER_TTS_KEY:
        return JSONResponse({"error": "Server chưa cấu hình OPENROUTER_TTS_KEY."}, status_code=400)

    # Giới hạn độ dài để tránh request quá nặng
    text = text[:4000]

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENROUTER_TTS_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "fish-audio/s2.1-pro-free:free",
                "input": text,
                "voice": "alloy",
            },
            timeout=60,
        )
        resp.raise_for_status()
        return FastAPIResponse(
            content=resp.content,
            media_type=resp.headers.get("Content-Type", "audio/mpeg"),
        )
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return JSONResponse({"error": f"Lỗi TTS: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi TTS: {str(e)}"}, status_code=502)


# =====================================================================
# Health check
# =====================================================================
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
