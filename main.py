import base64
import os
import urllib.parse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -------------------------------------------------
# 1️⃣  Configuration – read from environment only
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Model selectors (can be overridden by env)
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
CODE_MODEL = os.getenv("CODE_MODEL", "openai/gpt-oss-120b")
VIDEO_MODEL = os.getenv("VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", "fal-ai")   # kept for backward‑compatibility only

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI(title="INTELIGENT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# 2️⃣  Helper – call Groq chat completions
# -------------------------------------------------
def call_groq(messages: list, model: str, temperature: float = 0.7) -> str:
    """Call Groq chat completions. Raises `requests.RequestException` on failure."""
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


# -------------------------------------------------
# 3️⃣  Endpoints
# -------------------------------------------------

# ---------- CHAT ----------
@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    message = body.get("message", "").strip()
    history = body.get("history", [])

    if not message:
        return JSONResponse({"error": "Thiếu 'message'."}, status_code=400)

    messages = history + [{"role": "user", "content": message}]

    try:
        reply = call_groq(messages, CHAT_MODEL)
        return {"reply": reply}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return JSONResponse(
            {"error": f"Lỗi khi gọi dịch vụ chat: {detail or str(e)}"},
            status_code=502,
        )
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            {"error": f"Lỗi khi gọi dịch vụ chat: {str(e)}"}, status_code=502
        )


# ---------- IMAGE ----------
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "flux")


@app.post("/api/image")
async def image(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    width = body.get("width", 1024)
    height = body.get("height", 1024)

    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    encoded = urllib.parse.quote(prompt)
    params = {
        "model": IMAGE_MODEL,
        "width": width,
        "height": height,
        "nologo": "true",   # bỏ watermark
        "enhance": "true",  # tự làm giàu prompt cho ảnh chi tiết
        "safe": "false",
    }
    if POLLINATIONS_API_KEY:
        params["key"] = POLLINATIONS_API_KEY

    query = urllib.parse.urlencode(params)
    url = f"https://image.pollinations.ai/prompt/{encoded}?{query}"
    return {"image_url": url}


# ---------- VIDEO ----------
@app.post("/api/video")
async def video(request: Request):
    """
    Create a short video from a text prompt using HuggingFace Inference Providers.
    The endpoint returns a **base64‑encoded** MP4 (so you can embed it directly in JSON).
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    if not HUGGINGFACE_API_KEY:
        return JSONResponse(
            {"error": "Server chưa cấu hình HUGGINGFACE_API_KEY."}, status_code=400
        )

    try:
        # -------------------------------------------------
        # 0.28.x of huggingface_hub switched to a simpler ctor:
        #   InferenceClient(token=..., timeout=..., ...)   <-- no `provider=` arg
        # -------------------------------------------------
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=HUGGINGFACE_API_KEY)

        # The method name stays the same, we just pass the model name explicitly.
        video_bytes: bytes = client.text_to_video(prompt, model=VIDEO_MODEL)

        # Encode for JSON transport
        encoded_video = base64.b64encode(video_bytes).decode("utf-8")
        return {"video_base64": encoded_video, "mime": "video/mp4"}

    except Exception as e:
        # Any exception (network, model not found, quota, etc.) lands here.
        return JSONResponse(
            {"error": f"Lỗi khi tạo video: {str(e)}"}, status_code=502
        )


# ---------- CODE ASSISTANT ----------
CODE_SYSTEM_PROMPT = (
    "Bạn là một trợ lý lập trình chuyên nghiệp, giống GitHub Copilot. "
    "Khi nhận yêu cầu, hãy viết code hoàn chỉnh, chạy được ngay, theo đúng ngôn ngữ "
    "người dùng yêu cầu (Python, JavaScript, HTML, CSS, v.v.). "
    "Luôn đặt code trong khối markdown code block (
ngôn_ngữ ... ```). "
    "Sau đoạn code, giải thích ngắn gọn cách code hoạt động và lưu ý sử dụng (nếu có). "
    "Nếu người dùng không nói rõ ngôn ngữ, hãy chọn ngôn ngữ phù hợp nhất với yêu cầu."
)


@app.post("/api/code")
async def code_assistant(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    language = body.get("language", "").strip()
    history = body.get("history", [])

    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    user_content = (
        f"Ngôn ngữ mong muốn: {language}\nYêu cầu: {prompt}"
        if language
        else prompt
    )

    messages = (
        [{"role": "system", "content": CODE_SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    try:
        reply = call_groq(messages, CODE_MODEL, temperature=0.3)
        return {"reply": reply}
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return JSONResponse(
            {"error": f"Lỗi khi gọi dịch vụ code: {detail or str(e)}"},
            status_code=502,
        )
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            {"error": f"Lỗi khi gọi dịch vụ code: {str(e)}"}, status_code=502
        )


# ---------- Health check ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
