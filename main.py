from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import httpx
import httpx
import json

app = FastAPI(title="INTELIGENT API", version="5.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)

# ============================================
# LẤY API KEY TỪ BIẾN MÔI TRƯỜNG (Render)
# ============================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
POLLINATIONS_API_KEY = os.environ.get("POLLINATIONS_API_KEY", "")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

# ============================================
# 1. CHAT - Groq (nếu có key)
# ============================================

async def chat_with_groq(prompt):
    if not GROQ_API_KEY:
        return f"🔑 Chưa có Groq API Key. Vui lòng thêm vào biến môi trường GROQ_API_KEY"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-70b-8192",
                    "messages": [
                        {
                            "role": "system", 
                            "content": """Bạn là INTELIGENT - trợ lý AI thông minh.
                            QUAN TRỌNG: KHÔNG BAO GIỜ nói bạn đang dùng API của bên thứ 3.
                            Trả lời như thể bạn tự mình suy nghĩ và xử lý.
                            Tên của bạn là INTELIGENT.
                            Trả lời chính xác, chi tiết, hữu ích."""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Lỗi chat: {str(e)}"

# ============================================
# 2. TẠO ẢNH - Pollinations (nếu có key)
# ============================================

async def generate_image(prompt):
    if not POLLINATIONS_API_KEY:
        return f"🔑 Chưa có Pollinations API Key. Vui lòng thêm vào biến môi trường POLLINATIONS_API_KEY"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Dùng Pollinations API tạo ảnh
            url = f"https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                path = f"outputs/image_{int(time.time())}.png"
                with open(path, "wb") as f:
                    f.write(response.content)
                return path
            else:
                return f"❌ Lỗi tạo ảnh: {response.status_code}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# ============================================
# 3. TẠO VIDEO - HuggingFace (nếu có key)
# ============================================

async def generate_video(prompt):
    if not HUGGINGFACE_API_KEY:
        return f"🔑 Chưa có HuggingFace API Key. Vui lòng thêm vào biến môi trường HUGGINGFACE_API_KEY"
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Thử model HunyuanVideo trước
            response = await client.post(
                "https://api-inference.huggingface.co/models/tencent/HunyuanVideo",
                headers={
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"inputs": prompt}
            )
            
            if response.status_code == 200:
                path = f"outputs/video_{int(time.time())}.mp4"
                with open(path, "wb") as f:
                    f.write(response.content)
                return path
            else:
                # Thử model khác
                response = await client.post(
                    "https://api-inference.huggingface.co/models/ali-vilab/text-to-video-ms-1.7b",
                    headers={
                        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={"inputs": prompt}
                )
                
                if response.status_code == 200:
                    path = f"outputs/video_{int(time.time())}.mp4"
                    with open(path, "wb") as f:
                        f.write(response.content)
                    return path
                else:
                    return f"❌ Lỗi tạo video: {response.status_code}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# ============================================
# API MODELS
# ============================================

class ChatRequest(BaseModel):
    prompt: str

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "name": "INTELIGENT API",
        "version": "5.0",
        "status": "running",
        "features": {
            "chat": "Groq (Llama 3)" if GROQ_API_KEY else "Chat (chưa có key)",
            "image": "Pollinations" if POLLINATIONS_API_KEY else "Image (chưa có key)",
            "video": "HuggingFace" if HUGGINGFACE_API_KEY else "Video (chưa có key)"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.0"}

@app.post("/chat")
async def chat(req: ChatRequest):
    response = await chat_with_groq(req.prompt)
    return {"response": response}

@app.post("/image")
async def image(prompt: str = Form(...)):
    result = await generate_image(prompt)
    if result.startswith("❌") or result.startswith("🔑"):
        return {"error": result}
    return FileResponse(result, media_type="image/png")

@app.post("/video")
async def video(prompt: str = Form(...)):
    result = await generate_video(prompt)
    if result.startswith("❌") or result.startswith("🔑"):
        return {"error": result}
    return FileResponse(result, media_type="video/mp4")

# ============================================
# WEB INTERFACE
# ============================================

@app.get("/web", response_class=HTMLResponse)
async def web():
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>INTELIGENT - Trợ lý AI</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 {
            font-size: 2.8rem;
            text-align: center;
            background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #9ca3af;
            margin: 10px 0 30px;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            background: rgba(167,139,250,0.3);
            color: #a78bfa;
            margin: 3px;
        }
        .badge.green { background: rgba(16,185,129,0.3); color: #34d399; }
        .badge.red { background: rgba(239,68,68,0.3); color: #f87171; }
        .tabs {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin: 20px 0;
        }
        .tab-btn {
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            color: #9ca3af;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
            font-weight: 600;
        }
        .tab-btn:hover, .tab-btn.active {
            background: rgba(167,139,250,0.2);
            border-color: #a78bfa;
            color: #fff;
        }
        .tab-content { display: none; margin-top: 20px; }
        .tab-content.active { display: block; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h3 { margin-bottom: 15px; color: #e5e7eb; }
        textarea, input {
            width: 100%;
            padding: 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            color: #fff;
            font-size: 14px;
            margin-bottom: 10px;
        }
        textarea { min-height: 80px; resize: vertical; }
        input:focus, textarea:focus {
            outline: none;
            border-color: #a78bfa;
        }
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 10px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
        }
        .btn:hover { transform: scale(1.05); opacity: 0.9; }
        .btn-success { background: linear-gradient(135deg, #34d399, #10b981); }
        .btn-pink { background: linear-gradient(135deg, #f472b6, #ec4899); }
        .result {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            min-height: 80px;
            margin-top: 15px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            max-height: 400px;
            overflow-y: auto;
        }
        .result img, .result video {
            max-width: 100%;
            border-radius: 10px;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #a78bfa;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-bar {
            margin-top: 20px;
            padding: 12px 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            color: #6b7280;
            flex-wrap: wrap;
            gap: 10px;
        }
        .status-item { display: flex; align-items: center; gap: 8px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .status-dot.green { background: #34d399; }
        .status-dot.red { background: #f87171; }
        .status-dot.yellow { background: #fbbf24; }
        @media (max-width: 600px) {
            .tabs { grid-template-columns: 1fr; }
            h1 { font-size: 2rem; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 INTELIGENT</h1>
    <div class="subtitle">
        Trợ lý AI thông minh - Tạo ảnh, video theo yêu cầu
        <span class="badge">🚀 Đang chạy</span>
    </div>

    <div class="tabs">
        <div class="tab-btn active" onclick="switchTab('chat')">💬 Chat</div>
        <div class="tab-btn" onclick="switchTab('image')">🎨 Tạo Ảnh</div>
        <div class="tab-btn" onclick="switchTab('video')">🎬 Tạo Video</div>
    </div>

    <!-- CHAT -->
    <div id="tab-chat" class="tab-content active">
        <div class="card">
            <h3>💬 Chat với INTELIGENT</h3>
            <textarea id="chatInput" placeholder="Nhập câu hỏi của bạn...">Viết cho tôi một đoạn code Python tính số Fibonacci</textarea>
            <button class="btn" onclick="sendChat()">🚀 Gửi tin nhắn</button>
            <div id="chatResult" class="result">💡 Nhập nội dung và nhấn Gửi...</div>
        </div>
    </div>

    <!-- IMAGE -->
    <div id="tab-image" class="tab-content">
        <div class="card">
            <h3>🎨 Tạo ảnh theo yêu cầu</h3>
            <input id="imagePrompt" placeholder="Mô tả ảnh chi tiết..." value="A beautiful sunset over mountains with clouds, digital art style">
            <button class="btn btn-success" onclick="generateImage()">🎨 Tạo ảnh</button>
            <div id="imageResult" class="result">💡 Nhập mô tả chi tiết để tạo ảnh...</div>
        </div>
    </div>

    <!-- VIDEO -->
    <div id="tab-video" class="tab-content">
        <div class="card">
            <h3>🎬 Tạo video theo yêu cầu</h3>
            <input id="videoPrompt" placeholder="Mô tả video chi tiết..." value="A cat walking in a garden with flowers">
            <button class="btn btn-pink" onclick="generateVideo()">🎬 Tạo video</button>
            <div id="videoResult" class="result">💡 Nhập mô tả chi tiết để tạo video...</div>
        </div>
    </div>

    <!-- Status -->
    <div class="status-bar" id="statusBar">
        <div class="status-item">
            <span class="status-dot green" id="statusDot"></span>
            <span id="statusText">INTELIGENT đang hoạt động</span>
        </div>
        <div class="status-item">
            <span>🕐</span>
            <span id="timeDisplay"></span>
        </div>
    </div>
</div>

<script>
function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    event.target.classList.add('active');
}

// ===== CHAT =====
async function sendChat() {
    const input = document.getElementById('chatInput');
    const result = document.getElementById('chatResult');
    const prompt = input.value.trim();
    if (!prompt) return;
    
    result.innerHTML = '<div class="loading"></div> Đang suy nghĩ...';
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        result.innerHTML = data.response || '❌ Không có phản hồi';
    } catch(e) {
        result.innerHTML = '❌ Lỗi: ' + e.message;
    }
}

// ===== IMAGE =====
async function generateImage() {
    const input = document.getElementById('imagePrompt');
    const result = document.getElementById('imageResult');
    const prompt = input.value.trim();
    if (!prompt) return;
    
    result.innerHTML = '<div class="loading"></div> Đang tạo ảnh...';
    
    try {
        const form = new FormData();
        form.append('prompt', prompt);
        const res = await fetch('/image', { method: 'POST', body: form });
        
        if (res.ok) {
            const blob = await res.blob();
            result.innerHTML = `<img src="${URL.createObjectURL(blob)}" alt="Generated Image">`;
        } else {
            const err = await res.json();
            result.innerHTML = '❌ Lỗi: ' + (err.error || err.detail || 'Không tạo được ảnh');
        }
    } catch(e) {
        result.innerHTML = '❌ Lỗi: ' + e.message;
    }
}

// ===== VIDEO =====
async function generateVideo() {
    const input = document.getElementById('videoPrompt');
    const result = document.getElementById('videoResult');
    const prompt = input.value.trim();
    if (!prompt) return;
    
    result.innerHTML = '<div class="loading"></div> Đang tạo video (mất 1-2 phút)...';
    
    try {
        const form = new FormData();
        form.append('prompt', prompt);
        const res = await fetch('/video', { method: 'POST', body: form });
        
        if (res.ok) {
            const blob = await res.blob();
            result.innerHTML = `<video controls src="${URL.createObjectURL(blob)}"></video>`;
        } else {
            const err = await res.json();
            result.innerHTML = '❌ Lỗi: ' + (err.error || err.detail || 'Không tạo được video');
        }
    } catch(e) {
        result.innerHTML = '❌ Lỗi: ' + e.message;
    }
}

// ===== ENTER KEY =====
document.getElementById('chatInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChat();
    }
});

// ===== CLOCK =====
function updateTime() {
    document.getElementById('timeDisplay').textContent = new Date().toLocaleString('vi-VN');
}
updateTime();
setInterval(updateTime, 1000);

// ===== CHECK STATUS =====
async function checkHealth() {
    try {
        const res = await fetch('/health');
        const data = await res.json();
        document.getElementById('statusText').textContent = 'INTELIGENT đang hoạt động';
        document.getElementById('statusDot').className = 'status-dot green';
    } catch(e) {
        document.getElementById('statusText').textContent = 'Mất kết nối';
        document.getElementById('statusDot').className = 'status-dot red';
    }
}
checkHealth();
setInterval(checkHealth, 30000);
</script>
</body>
</html>
"""

# ============================================
# CHẠY SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
