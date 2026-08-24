from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import base64
import time

app = FastAPI(title="INTELIGENT API")

# Serve static files (logo etc.) from ./static if present
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Read API keys from environment (never hardcoded)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Helper: simple HuggingFace text generation call (deferred, only on request)
def hf_text_generation(prompt: str, model: str = "gpt2", max_tokens: int = 150):
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY is not set")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "do_sample": False}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace text error {resp.status_code}: {resp.text}")
    try:
        data = resp.json()
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)
    except Exception:
        return resp.text

# Helper: HuggingFace image inference -> return data URL (base64 PNG)
def hf_image_generation(prompt: str, model: str = "stabilityai/stable-diffusion-2"):
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY is not set")
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Accept": "application/json, image/png"}
    payload = {"inputs": prompt}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code == 503:
        # model might be loading
        time.sleep(1.0)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace image error {resp.status_code}: {resp.text}")
    ct = resp.headers.get("content-type", "")
    if "image" in ct:
        img_bytes = resp.content
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{ct};base64,{b64}"
    # fallback: try parse json fields
    try:
        j = resp.json()
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
    # Serve index.html if exists
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"status": "ok", "msg": "index.html not found"})

@app.post("/api/chat")
async def api_chat(req: Request):
    body = await req.json()
    message = body.get("message") or body.get("prompt") or ""
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' in request")
    try:
        reply = hf_text_generation(message, model="gpt2", max_tokens=200)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.post("/api/image")
async def api_image(req: Request):
    body = await req.json()
    prompt = body.get("prompt") or ""
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request")
    try:
        img = hf_image_generation(prompt)
        return {"image": img}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

@app.post("/api/video")
async def api_video(req: Request):
    body = await req.json()
    prompt = body.get("prompt") or ""
    frames = int(body.get("frames") or 6)
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'")
    if frames < 2 or frames > 40:
        raise HTTPException(status_code=400, detail="frames must be 2..40")
    frames_list = []
    for i in range(frames):
        try:
            frame_prompt = f"{prompt} --frame {i+1}/{frames}"
            img = hf_image_generation(frame_prompt)
            frames_list.append(img)
            time.sleep(0.6)  # avoid hitting rate limits too fast
        except Exception as e:
            return JSONResponse({"frames": frames_list, "error": str(e)}, status_code=207)
    return {"frames": frames_list}

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "hf_key_set": bool(HUGGINGFACE_API_KEY),
        "pollinations_key_set": bool(POLLINATIONS_API_KEY),
        "groq_key_set": bool(GROQ_API_KEY),
    }
