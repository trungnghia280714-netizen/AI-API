import asyncio
import base64
import json
import os
import re
import time
import urllib.parse
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
# Mỗi biến có thể chứa NHIỀU key cách nhau bằng dấu phẩy, vd:
# DEEPSEEK_API_KEY=key1,key2,key3
# -> khi 1 key bị giới hạn (429) hoặc lỗi xác thực, tự động thử key kế tiếp.
def _parse_keys(env_name: str) -> list:
    raw = os.environ.get(env_name, "")
    return [k.strip() for k in raw.split(",") if k.strip()]

NVIDIA_API_KEYS = _parse_keys("NVIDIA_API_KEY")         # Chat (NVIDIA NIM - free tier khá rộng)
ANTHROPIC_API_KEYS = _parse_keys("ANTHROPIC_API_KEY")   # Code + Vision (Claude)
OPENAI_API_KEYS = _parse_keys("OPENAI_API_KEY")         # Ảnh
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")   # Video (Veo) - chưa xoay vòng

CHAT_MODEL = os.environ.get("CHAT_MODEL", "deepseek-ai/deepseek-v4-pro-0813")
CODE_MODEL = os.environ.get("CODE_MODEL", "claude-sonnet-5")
VISION_MODEL = os.environ.get("VISION_MODEL", "claude-sonnet-5")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gpt-image-2")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "veo-3.1-lite-generate-preview")

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Hạn mức gói free / ngày (theo tài khoản đã đăng nhập)
DAILY_LIMITS = {"chat": 40, "code": 5, "image": 5, "video": 1}
FEATURE_NAMES_VI = {"chat": "Chat", "code": "Code Assistant", "image": "Tạo ảnh", "video": "Tạo video"}
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
# Hàm gọi các AI provider trả phí
# =====================================================================
def call_nvidia(messages: list, model: str, temperature: float = 0.7):
    """Chat - NVIDIA NIM, API kiểu OpenAI-compatible (build.nvidia.com).
    Tự động xoay vòng qua các key trong NVIDIA_API_KEYS nếu 1 key bị lỗi 429/401."""
    if not NVIDIA_API_KEYS:
        raise ValueError("Server chưa cấu hình NVIDIA_API_KEY.")

    last_error = None
    for key in NVIDIA_API_KEYS:
        try:
            resp = requests.post(
                NVIDIA_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "max_tokens": 4096,
                },
                timeout=60,
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


def call_claude(system_prompt: str, messages: list, model: str, max_tokens: int = 4096):
    """Code + Vision - Claude API (Anthropic). messages theo format Anthropic Messages API
    (content có thể là string hoặc list block cho vision).
    Tự động xoay vòng qua các key trong ANTHROPIC_API_KEYS nếu 1 key bị lỗi 429/401."""
    if not ANTHROPIC_API_KEYS:
        raise ValueError("Server chưa cấu hình ANTHROPIC_API_KEY.")

    body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system_prompt:
        body["system"] = system_prompt

    last_error = None
    for key in ANTHROPIC_API_KEYS:
        try:
            resp = requests.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            text_parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
            return "".join(text_parts)
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code in (401, 429):
                continue
            raise
    raise last_error


def extract_claude_error(e: requests.exceptions.HTTPError) -> str:
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
# 1. CHAT (DeepSeek)
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

    if not message:
        return JSONResponse({"error": "Thiếu 'message'."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "chat")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    messages = history + [{"role": "user", "content": message}]

    try:
        reply = call_nvidia(messages, CHAT_MODEL)
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
# 1b. VISION (Claude - hỏi AI về nội dung ảnh đính kèm)
# =====================================================================
@app.post("/api/vision")
async def vision(
    request: Request,
    user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip() or "Mô tả và phân tích nội dung ảnh này."
    image_base64 = body.get("image_base64", "")
    mime = body.get("mime", "image/jpeg")
    conversation_id = body.get("conversation_id")

    if not image_base64:
        return JSONResponse({"error": "Thiếu 'image_base64'."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "chat")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": image_base64},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        reply = call_claude("", messages, VISION_MODEL)
        result = {"reply": reply}
        if user:
            result["conversation_id"] = save_turn(
                db, user, conversation_id, "chat", f"[Ảnh đính kèm] {prompt}", reply
            )
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = extract_claude_error(e)
        return JSONResponse({"error": f"Lỗi khi phân tích ảnh: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi phân tích ảnh: {str(e)}"}, status_code=502)


# =====================================================================
# 2. ẢNH (OpenAI - GPT Image 2)
# =====================================================================
@app.post("/api/image")
async def image(
    request: Request,
    user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    if not OPENAI_API_KEYS:
        return JSONResponse({"error": "Server chưa cấu hình OPENAI_API_KEY."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "image")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    last_error = None
    for key in OPENAI_API_KEYS:
        try:
            resp = requests.post(
                OPENAI_IMAGE_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": IMAGE_MODEL, "prompt": prompt, "size": "1024x1024", "n": 1},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            if not items or not items[0].get("b64_json"):
                return JSONResponse({"error": "Không nhận được ảnh từ OpenAI."}, status_code=502)
            return {"image_base64": items[0]["b64_json"], "mime": "image/png"}
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code in (401, 429):
                continue  # thử key kế tiếp
            detail = extract_openai_style_error(e)
            return JSONResponse({"error": f"Lỗi khi gọi dịch vụ ảnh: {detail or str(e)}"}, status_code=502)
        except requests.exceptions.RequestException as e:
            return JSONResponse({"error": f"Lỗi khi gọi dịch vụ ảnh: {str(e)}"}, status_code=502)

    detail = extract_openai_style_error(last_error) if last_error else ""
    return JSONResponse(
        {"error": f"Tất cả key OpenAI đều bị giới hạn hoặc lỗi: {detail or str(last_error)}"},
        status_code=502,
    )


# =====================================================================
# 3. VIDEO (Google Veo 3.1 qua Gemini API)
# =====================================================================
def _generate_veo_video_sync(prompt: str):
    """Chạy đồng bộ (blocking) trong thread riêng - gọi Veo, poll đến khi xong, tải video về.
    Trả về (video_bytes, error_message). Chỉ 1 trong 2 giá trị khác None."""
    if not GEMINI_API_KEY:
        return None, "Server chưa cấu hình GEMINI_API_KEY."

    try:
        start_resp = requests.post(
            f"{GEMINI_BASE_URL}/models/{VIDEO_MODEL}:predictLongRunning",
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json={"instances": [{"prompt": prompt}]},
            timeout=30,
        )
        start_resp.raise_for_status()
        operation_name = start_resp.json().get("name")
        if not operation_name:
            return None, "Không nhận được operation từ Veo."

        max_wait_seconds = 280
        interval_seconds = 10
        elapsed = 0
        status_data = None

        while elapsed < max_wait_seconds:
            time.sleep(interval_seconds)
            elapsed += interval_seconds
            status_resp = requests.get(
                f"{GEMINI_BASE_URL}/{operation_name}",
                headers={"x-goog-api-key": GEMINI_API_KEY},
                timeout=30,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            if status_data.get("done"):
                break
        else:
            return None, "Tạo video quá lâu (vượt quá 4.5 phút), vui lòng thử lại."

        if status_data.get("error"):
            return None, status_data["error"].get("message", "Lỗi không xác định từ Veo.")

        samples = (
            status_data.get("response", {})
            .get("generateVideoResponse", {})
            .get("generatedSamples", [])
        )
        if not samples:
            return None, "Veo không trả về video nào (có thể bị chặn bởi bộ lọc an toàn)."

        video_uri = samples[0]["video"]["uri"]
        video_resp = requests.get(
            video_uri, headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=120
        )
        video_resp.raise_for_status()
        return video_resp.content, None

    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return None, detail or str(e)
    except requests.exceptions.RequestException as e:
        return None, str(e)


@app.post("/api/video")
async def video(
    request: Request,
    user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "video")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    video_bytes, error = await asyncio.to_thread(_generate_veo_video_sync, prompt)

    if error:
        return JSONResponse({"error": f"Lỗi khi tạo video: {error}"}, status_code=502)

    encoded_video = base64.b64encode(video_bytes).decode("utf-8")
    return {"video_base64": encoded_video, "mime": "video/mp4"}


# =====================================================================
# 4. CODE ASSISTANT (Claude)
# =====================================================================
CODE_SYSTEM_PROMPT = (
    "Bạn là một trợ lý lập trình chuyên nghiệp, giống GitHub Copilot. "
    "Khi nhận yêu cầu, hãy viết code hoàn chỉnh, chạy được ngay, theo đúng ngôn ngữ "
    "người dùng yêu cầu (Python, JavaScript, HTML, CSS, v.v.). "
    "Luôn đặt code trong khối markdown code block (```ngôn_ngữ ... ```). "
    "Sau đoạn code, giải thích ngắn gọn cách code hoạt động và lưu ý sử dụng (nếu có). "
    "Nếu người dùng không nói rõ ngôn ngữ, hãy chọn ngôn ngữ phù hợp nhất với yêu cầu."
)


@app.post("/api/code")
async def code_assistant(
    request: Request,
    user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    language = body.get("language", "").strip()
    history = body.get("history", [])
    conversation_id = body.get("conversation_id")

    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "code")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    user_content = f"Ngôn ngữ mong muốn: {language}\nYêu cầu: {prompt}" if language else prompt

    messages = history + [{"role": "user", "content": user_content}]

    try:
        reply = call_claude(CODE_SYSTEM_PROMPT, messages, CODE_MODEL)
        result = {"reply": reply}
        if user:
            result["conversation_id"] = save_turn(db, user, conversation_id, "code", prompt, reply)
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = extract_claude_error(e)
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {str(e)}"}, status_code=502)


# ---------- Health check ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
