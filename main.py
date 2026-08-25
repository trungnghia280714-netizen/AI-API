import base64
import os
import urllib.parse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------- Cấu hình từ biến môi trường (KHÔNG hardcode key) ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

# Groq đã khai tử llama-3.3-70b-versatile (6/2026) -> dùng model thay thế
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b")
CODE_MODEL = os.environ.get("CODE_MODEL", "openai/gpt-oss-120b")
# HuggingFace: model + provider cho text-to-video qua Inference Providers (router mới)
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
VIDEO_PROVIDER = os.environ.get("VIDEO_PROVIDER", "fal-ai")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = FastAPI(title="INTELIGENT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def call_groq(messages: list, model: str, temperature: float = 0.7):
    """Gọi Groq chat completions, ném lỗi requests.RequestException nếu fail."""
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


# ---------- 1. CHAT ----------
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
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {str(e)}"}, status_code=502)


# ---------- 2. ẢNH ----------
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "flux")

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
        "model": IMAGE_MODEL,   # flux cho chất lượng tốt hơn model mặc định cũ
        "width": width,
        "height": height,
        "nologo": "true",       # bỏ watermark
        "enhance": "true",      # tự làm giàu prompt cho ảnh chi tiết, đúng ý hơn
        "safe": "false",
    }
    if POLLINATIONS_API_KEY:
        params["key"] = POLLINATIONS_API_KEY

    query = urllib.parse.urlencode(params)
    url = f"https://image.pollinations.ai/prompt/{encoded}?{query}"

    return {"image_url": url}


# ---------- 3. VIDEO ----------
@app.post("/api/video")
async def video(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    if not HUGGINGFACE_API_KEY:
        return JSONResponse({"error": "Server chưa cấu hình HUGGINGFACE_API_KEY."}, status_code=400)

    try:
        # HuggingFace đã chuyển sang hệ thống Inference Providers (router.huggingface.co).
        # Dùng SDK chính thức để xử lý routing + polling cho đúng.
        from huggingface_hub import InferenceClient

        client = InferenceClient(provider=VIDEO_PROVIDER, api_key=HUGGINGFACE_API_KEY)
        video_bytes = client.text_to_video(prompt, model=VIDEO_MODEL)

        encoded_video = base64.b64encode(video_bytes).decode("utf-8")
        return {"video_base64": encoded_video, "mime": "video/mp4"}

    except Exception as e:
        return JSONResponse({"error": f"Lỗi khi tạo video: {str(e)}"}, status_code=502)


# ---------- 4. CODE ASSISTANT ----------
CODE_SYSTEM_PROMPT = (
    "Bạn là một trợ lý lập trình chuyên nghiệp, giống GitHub Copilot. "
    "Khi nhận yêu cầu, hãy viết code hoàn chỉnh, chạy được ngay, theo đúng ngôn ngữ "
    "người dùng yêu cầu (Python, JavaScript, HTML, CSS, v.v.). "
    "Luôn đặt code trong khối markdown code block (```ngôn_ngữ ... ```). "
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

    user_content = f"Ngôn ngữ mong muốn: {language}\nYêu cầu: {prompt}" if language else prompt

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
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {str(e)}"}, status_code=502)


# ---------- Health check ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
