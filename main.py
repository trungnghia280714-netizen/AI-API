from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import base64
import time

app = FastAPI(title="INTELIGENT API")

# CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Khi deploy, bạn có thể giới hạn domain ở đây
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đọc API keys từ environment (KHÔNG hardcode)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Helper: call HuggingFace text model for chat
def hf_text_generation(prompt: str, model: str = "gpt2", max_tokens: int = 150):
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY is not set in environment")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "do_sample": False}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace text generation error: {resp.status_code} {resp.text}")
    try:
        data = resp.json()
        # many HF text models return [{"generated_text": "..."}] or {"generated_text":"..."}
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # fallback: return string
        return str(data)
    except ValueError:
        return resp.text

# Helper: call HuggingFace image model (stable-diffusion) and return base64png
def hf_image_generation(prompt: str, model: str = "stabilityai/stable-diffusion-2", wait: float = 0.5):
    """
    Tries to use the HuggingFace Inference API to generate an image. Returns a data URL (base64 PNG).
    If the model or key is not available, raises RuntimeError.
    """
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY is not set in environment")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Accept": "application/json, image/png"
    }
    payload = {"inputs": prompt}
    # Some HF models return image bytes directly
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 503:
        # model loading - wait and retry once
        time.sleep(wait)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        # try to provide helpful error
        raise RuntimeError(f"HuggingFace image generation error: {resp.status_code} {resp.text}")

    ct = resp.headers.get("content-type", "")
    if "image" in ct:
        img_bytes = resp.content
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:{ct};base64,{b64}"
        return data_url
    else:
        # sometimes HF returns JSON with an error or with base64 field
        try:
            j = resp.json()
            # if API returns {'image_base64': '...'} etc.
            for k in ("image_base64", "b64_json", "image"):
                if k in j:
                    b64 = j[k]
                    if not b64.startswith("data:"):
                        return "data:image/png;base64," + b64
                    return b64
            return str(j)
        except Exception:
            return resp.text

@app.get("/")
async def root():
    # Serve the frontend index.html (must exist in the same directory)
    return FileResponse("index.html")

@app.post("/api/chat")
async def api_chat(req: Request):
    payload = await req.json()
    message = payload.get("message") or payload.get("prompt") or ""
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' in JSON body")
    try:
        # Use HuggingFace model for chat (fallback: gpt2). You can change the model name if you have hosted a chat model.
        reply = hf_text_generation(message, model="gpt2", max_tokens=200)
        return JSONResponse({"reply": reply})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image")
async def api_image(req: Request):
    """
    Body JSON: { "prompt": "a fantasy castle made of clouds" }
    Returns: { "image": "data:image/png;base64,..." }
    """
    payload = await req.json()
    prompt = payload.get("prompt") or ""
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in JSON body")
    try:
        # First attempt: use Pollinations if POLLINATIONS_API_KEY is provided (optional)
        # Note: Pollinations public endpoint also accepts image generation via simple GET, but here we prefer HF.
        image_data = hf_image_generation(prompt)
        return JSONResponse({"image": image_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

@app.post("/api/video")
async def api_video(req: Request):
    """
    Body JSON: { "prompt": "...", "frames": 8 }
    Server will generate `frames` images (calls the image generator) and return a JSON array of base64 images.
    The frontend will assemble them into a video.
    """
    payload = await req.json()
    prompt = payload.get("prompt") or ""
    frames = int(payload.get("frames") or 8)
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in JSON body")
    if frames < 2 or frames > 40:
        raise HTTPException(status_code=400, detail="frames must be between 2 and 40")
    results = []
    for i in range(frames):
        try:
            # Slightly vary prompt to encourage variation across frames
            frame_prompt = f"{prompt} --frame {i+1} of {frames}"
            img = hf_image_generation(frame_prompt)
            results.append(img)
            # short delay to avoid throttling
            time.sleep(0.8)
        except Exception as e:
            # if one frame fails, return what we have with an error flag
            return JSONResponse({"frames": results, "error": f"frame {i} failed: {e}"}, status_code=207)
    return JSONResponse({"frames": results})

# Health
@app.get("/api/health")
async def health():
    return {"status": "ok", "hf_key_set": bool(HUGGINGFACE_API_KEY), "pollinations_key_set": bool(POLLINATIONS_API_KEY), "groq_key_set": bool(GROQ_API_KEY)}
