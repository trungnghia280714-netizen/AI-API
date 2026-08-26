import base64
import os
import urllib.parse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------- Cấu hình từ biến môi trường (KHÔNG hardcode key) ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

# Groq đã khai tử llama-3.3-70b-versatile (6/2026) -> dùng model thay thế
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-120b")
CODE_MODEL = os.environ.get("CODE_MODEL", "openai/gpt-oss-120b")
# Gemini "Nano Banana 2" - model tạo ảnh khuyến nghị hiện tại của Google (11/2026)
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "gemini-3.1-flash-image")
# HuggingFace: model + provider cho text-to-video qua Inference Providers (router mới)
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
VIDEO_PROVIDER = os.environ.get("VIDEO_PROVIDER", "fal-ai")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Google đã chuyển tạo ảnh sang "Interactions API" mới, khác hẳn generateContent cũ
GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

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


def extract_groq_error(e: requests.exceptions.HTTPError) -> str:
    try:
        return e.response.json().get("error", {}).get("message", "")
    except Exception:
        return ""


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
        detail = extract_groq_error(e)
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ chat: {str(e)}"}, status_code=502)


# ---------- 2. ẢNH (Gemini "Nano Banana 2" qua Interactions API) ----------
@app.post("/api/image")
async def image(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Body request không hợp lệ (cần JSON)."}, status_code=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse({"error": "Thiếu 'prompt'."}, status_code=400)

    if not GEMINI_API_KEY:
        return JSONResponse({"error": "Server chưa cấu hình GEMINI_API_KEY."}, status_code=400)

    try:
        resp = requests.post(
            GEMINI_INTERACTIONS_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": IMAGE_MODEL,
                "input": prompt,
                "response_format": {"type": "image"},
            },
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()

        # Cách 1: field tiện lợi output_image (ưu tiên)
        output_image = data.get("output_image")
        image_b64 = None
        mime = "image/png"

        if output_image and output_image.get("data"):
            image_b64 = output_image["data"]
            mime = output_image.get("mime_type", "image/png")
        else:
            # Cách 2: fallback duyệt qua steps -> content blocks kiểu "image"
            for step in data.get("steps", []):
                if step.get("type") != "model_output":
                    continue
                for block in step.get("content", []):
                    if block.get("type") == "image" and block.get("data"):
                        image_b64 = block["data"]
                        mime = block.get("mime_type", "image/png")
                        break
                if image_b64:
                    break

        if not image_b64:
            return JSONResponse({"error": "Gemini không trả về ảnh nào (có thể do bộ lọc an toàn)."}, status_code=502)

        return {"image_base64": image_b64, "mime": mime}

    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ ảnh: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ ảnh: {str(e)}"}, status_code=502)


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
        detail = extract_groq_error(e)
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {detail or str(e)}"}, status_code=502)
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": f"Lỗi khi gọi dịch vụ code: {str(e)}"}, status_code=502)


# ---------- Health check ----------
@app.get("/")
async def root():
    return {"status": "ok", "service": "INTELIGENT Backend"}
