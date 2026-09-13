import json
import os
import re
from datetime import date

import requests
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# ---------- Cấu hình từ biến môi trường (KHÔNG hardcode key) ----------
def _parse_keys(env_name: str) -> list:
    """Mỗi biến có thể chứa NHIỀU key cách nhau bằng dấu phẩy, vd: XKIRO_API_KEY=key1,key2
    -> khi 1 key bị giới hạn (429) hoặc lỗi xác thực, tự động thử key kế tiếp."""
    raw = os.environ.get(env_name, "")
    return [k.strip() for k in raw.split(",") if k.strip()]

XKIRO_API_KEYS = _parse_keys("XKIRO_API_KEY")
APINEX_API_KEYS = _parse_keys("APINEX_API_KEY")
OMNIROUTE_API_KEYS = _parse_keys("OMNIROUTE_API_KEYS")
CODECRAFT_API_KEYS = _parse_keys("CODECRAFT_API_KEY")   # <-- thêm mới

XKIRO_BASE_URL = os.environ.get("XKIRO_BASE_URL", "https://api.xkiro.com/v1")
APINEX_BASE_URL = os.environ.get("APINEX_BASE_URL", "https://apinex.bond/v1")
OMNIROUTE_BASE_URL = os.environ.get("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
CODECRAFT_BASE_URL = os.environ.get("CODECRAFT_BASE_URL", "https://codecraftapi.com/v1")  # <-- thêm mới

MODEL_CATALOG = {
    "deepseek-flash": {
        "label": "DeepSeek Flash",
        "url": f"{XKIRO_BASE_URL}/chat/completions",
        "keys": XKIRO_API_KEYS,
        "model": "deepseek/deepseek-v4-flash",
    },
    "deepseek-pro": {
        "label": "DeepSeek V4 Pro",
        "url": f"{XKIRO_BASE_URL}/chat/completions",
        "keys": XKIRO_API_KEYS,
        "model": "deepseek/deepseek-v4-pro",
    },
    "gemini-3-8-flash": {
        "label": "Gemini 3.8 Flash",
        "url": f"{APINEX_BASE_URL}/chat/completions",
        "keys": APINEX_API_KEYS,
        "model": "gemini-3.8-flash",
    },
    "omniroute-free": {
        "label": "OmniRoute Free",
        "url": f"{OMNIROUTE_BASE_URL}/chat/completions",
        "keys": OMNIROUTE_API_KEYS,
        "model": "openrouter/openrouter/free",
    },
    "claude-sonnet-5": {                                  # <-- thêm mới
        "label": "Claude Sonnet 5",
        "url": f"{CODECRAFT_BASE_URL}/chat/completions",
        "keys": CODECRAFT_API_KEYS,
        "model": "claude-sonnet-5",   # chỉnh lại đúng tên model mà codecraftapi.com yêu cầu
    },
}
DEFAULT_MODEL_ID = "deepseek-flash"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Hạn mức gói free / ngày (theo tài khoản đã đăng nhập)
DAILY_LIMITS = {"chat": 40}
FEATURE_NAMES_VI = {"chat": "Chat"}
UNLIMITED_PLANS = {"inteligent_cold", "inteligent_super_cold"}


def check_and_increment_usage(db: Session, user, feature: str):
    """Trả về None nếu còn hạn mức (đã tăng đếm), hoặc chuỗi lỗi nếu đã hết hạn mức hôm nay.
    Nếu user=None (khách chưa đăng nhập) hoặc user có gói không giới hạn -> bỏ qua, trả về None."""
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

    body = {"model": entry["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

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
            return data["choices"][0]["message"]["content"]
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


# ---------- Health check ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
