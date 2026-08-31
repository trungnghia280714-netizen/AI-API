import asyncio
import base64
import json
import os
import re
import time
import urllib.parse

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
from datetime import datetime, date

# ---------- Cấu hình từ biến môi trường (KHÔNG hardcode key) ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # dùng cho Veo 3.1 (tạo video)

CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b")
CODE_MODEL = os.environ.get("CODE_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3.6-27b")  # model hiểu ảnh của Groq
# Gemini image (Nano Banana 2) không có hạn mức free -> dùng Pollinations (model Flux, miễn phí)
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "flux")
# Veo 3.1 Lite: model video rẻ nhất của Google, dùng chung GEMINI_API_KEY
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "veo-3.1-lite-generate-preview")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
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


def call_groq(messages: list, model: str, temperature: float = 0.7):
    if not GROQ_API_KEY:
        raise ValueError("Server chưa cấu hình GROQ_API_KEY.")

    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def extract_groq_error(e: requests.exceptions.HTTPError) -> str:
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
            # Tài khoản mới đăng nhập lần đầu bằng Google -> tạo user không có mật khẩu
            user = User(email=email, password_hash=None)
            db.add(user)
            db.commit()
            db.refresh(user)

        token = create_access_token(user.id)
        return {"token": token, "email": user.email}

    except ValueError as e:
        return JSONResponse({"error": f"Xác thực Google thất bại: {str(e)}"}, status_code=401)
    except Exception as e:
        # Bắt mọi lỗi bất ngờ (vd: DB thiếu cột, mất kết nối...) để luôn trả JSON có CORS header,
        # tránh trình duyệt hiểu nhầm thành lỗi CORS khi server crash không kiểm soát.
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
# 1. CHAT
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
        reply = call_groq(messages, CHAT_MODEL)
        result = {"reply": reply}
        if user:
            result["conversation_id"] = save_turn(db, user, conversation_id, "chat", message, reply)
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = extract_groq_error(e)
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {str(e)}"}, status_code=502)


# =====================================================================
# 1b. VISION (hỏi AI về nội dung ảnh đính kèm - dùng model vision của Groq)
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

    if not GROQ_API_KEY:
        return JSONResponse({"error": "Server chưa cấu hình GROQ_API_KEY."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "chat")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_base64}"}},
            ],
        }
    ]

    try:
        reply = call_groq(messages, VISION_MODEL)
        result = {"reply": reply}
        if user:
            result["conversation_id"] = save_turn(
                db, user, conversation_id, "chat", f"[Ảnh đính kèm] {prompt}", reply
            )
        return result
    except requests.exceptions.HTTPError as e:
        detail = extract_groq_error(e)
        return JSONResponse({"error": f"Lỗi khi phân tích ảnh: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi phân tích ảnh: {str(e)}"}, status_code=502)


# =====================================================================
# 2. ẢNH (Pollinations - model Flux, miễn phí)
# =====================================================================
def translate_prompt_to_english(prompt: str) -> str:
    """Dịch/diễn giải prompt sang tiếng Anh chi tiết cho model Flux hiểu đúng ý hơn.
    Nếu Groq lỗi vì bất kỳ lý do gì, âm thầm fallback về prompt gốc."""
    if not GROQ_API_KEY:
        return prompt
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Dịch mô tả ảnh sau sang tiếng Anh, giữ đúng chủ thể và ý nghĩa gốc, "
                    "có thể bổ sung vài chi tiết hình ảnh (ánh sáng, góc chụp, phong cách) "
                    "để ảnh đẹp hơn nhưng KHÔNG được đổi chủ thể chính. "
                    "Chỉ trả về câu mô tả tiếng Anh, không giải thích gì thêm."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return call_groq(messages, CHAT_MODEL, temperature=0.4).strip()
    except Exception:
        return prompt


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
    width = body.get("width", 1024)
    height = body.get("height", 1024)

    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    usage_error = check_and_increment_usage(db, user, "image")
    if usage_error:
        return JSONResponse({"error": usage_error}, status_code=429)

    final_prompt = translate_prompt_to_english(prompt)
    encoded = urllib.parse.quote(final_prompt)

    params = {
        "model": IMAGE_MODEL,  # flux
        "width": width,
        "height": height,
        "nologo": "true",
        "safe": "false",
    }
    if POLLINATIONS_API_KEY:
        params["key"] = POLLINATIONS_API_KEY

    query = urllib.parse.urlencode(params)
    image_url = f"https://image.pollinations.ai/prompt/{encoded}?{query}"

    try:
        # Tải ảnh về rồi trả base64 cho frontend, để đồng nhất với các route khác
        # và tránh lộ trực tiếp URL công khai (có thể chứa key) ra trình duyệt.
        img_resp = requests.get(image_url, timeout=60)
        img_resp.raise_for_status()
        mime = img_resp.headers.get("content-type", "image/jpeg")
        image_b64 = base64.b64encode(img_resp.content).decode("utf-8")
        return {"image_base64": image_b64, "mime": mime}
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ ảnh: {str(e)}"}, status_code=502)


# =====================================================================
# 3. VIDEO (Google Veo 3.1 qua Gemini API - TRẢ PHÍ theo giây video)
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

        max_wait_seconds = 280  # ~4.5 phút, Veo có thể mất tới vài phút
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

    # Chạy trong thread riêng để không chặn event loop trong lúc poll (có thể mất vài phút)
    video_bytes, error = await asyncio.to_thread(_generate_veo_video_sync, prompt)

    if error:
        return JSONResponse({"error": f"Lỗi khi tạo video: {error}"}, status_code=502)

    encoded_video = base64.b64encode(video_bytes).decode("utf-8")
    return {"video_base64": encoded_video, "mime": "video/mp4"}


# =====================================================================
# 4. CODE ASSISTANT
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

    messages = (
        [{"role": "system", "content": CODE_SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    try:
        reply = call_groq(messages, CODE_MODEL, temperature=0.3)
        result = {"reply": reply}
        if user:
            result["conversation_id"] = save_turn(db, user, conversation_id, "code", prompt, reply)
        return result
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = extract_groq_error(e)
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {str(e)}"}, status_code=502)


# ---------- Health check ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
