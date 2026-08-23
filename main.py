from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time

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
# MOCK AI - Trả lời mẫu khi chưa có API key
# ============================================

def mock_chat(prompt):
    return f"INTELIGENT đã nhận: '{prompt}'\n\nĐây là câu trả lời mẫu. API đang hoạt động tốt!"

def mock_image(prompt):
    return "Mô tả ảnh của bạn: " + prompt

def mock_video(prompt):
    return "Mô tả video của bạn: " + prompt

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
        "message": "API đã được cấu hình để chạy trên Render Free. Hãy thay thế phần mock bằng logic thật của bạn."
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "5.0"}

@app.post("/chat")
async def chat(req: ChatRequest):
    response = mock_chat(req.prompt)
    return {"response": response}

@app.post("/image")
async def image(prompt: str = Form(...)):
    return {"response": mock_image(prompt), "status": "success"}

@app.post("/video")
async def video(prompt: str = Form(...)):
    return {"response": mock_video(prompt), "status": "success"}

@app.get("/web", response_class=HTMLResponse)
async def web():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>INTELIGENT - API Đang Chạy</title>
        <style>
            body { font-family: -apple-system, sans-serif; background: #0a0a0f; color: #fff; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.05); border-radius: 20px; padding: 30px; }
            h1 { text-align: center; background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .subtitle { text-align: center; color: #9ca3af; margin-bottom: 30px; }
            .card { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 20px 0; }
            textarea, input { width: 100%; padding: 12px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: #fff; margin-bottom: 10px; }
            .btn { padding: 12px 30px; border: none; border-radius: 10px; color: #fff; cursor: pointer; background: linear-gradient(135deg, #a78bfa, #60a5fa); }
            .result { background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px; margin-top: 15px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 INTELIGENT</h1>
            <div class="subtitle">✅ API đã được triển khai thành công trên Render!</div>
            
            <div class="card">
                <h3>💬 Chat</h3>
                <textarea id="chatInput" rows="3">Xin chào INTELIGENT!</textarea>
                <button class="btn" onclick="sendChat()">Gửi</button>
                <div id="chatResult" class="result">Nhập tin nhắn và nhấn Gửi...</div>
            </div>
            
            <div class="card">
                <h3>🎨 Tạo ảnh</h3>
                <input id="imagePrompt" value="một bức ảnh đẹp">
                <button class="btn" onclick="generateImage()">Tạo</button>
                <div id="imageResult" class="result">...</div>
            </div>
            
            <div class="card">
                <h3>🎬 Tạo video</h3>
                <input id="videoPrompt" value="một video hay">
                <button class="btn" onclick="generateVideo()">Tạo</button>
                <div id="videoResult" class="result">...</div>
            </div>
        </div>
        
        <script>
            async function sendChat() {
                const prompt = document.getElementById('chatInput').value;
                const result = document.getElementById('chatResult');
                result.innerHTML = '⏳ Đang xử lý...';
                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt})
                    });
                    const data = await res.json();
                    result.innerHTML = data.response || 'Không có phản hồi';
                } catch(e) {
                    result.innerHTML = '❌ Lỗi: ' + e.message;
                }
            }
            
            async function generateImage() {
                const prompt = document.getElementById('imagePrompt').value;
                const result = document.getElementById('imageResult');
                result.innerHTML = '⏳ Đang tạo ảnh...';
                try {
                    const form = new FormData();
                    form.append('prompt', prompt);
                    const res = await fetch('/image', {method: 'POST', body: form});
                    const data = await res.json();
                    result.innerHTML = data.response || 'Không có phản hồi';
                } catch(e) {
                    result.innerHTML = '❌ Lỗi: ' + e.message;
                }
            }
            
            async function generateVideo() {
                const prompt = document.getElementById('videoPrompt').value;
                const result = document.getElementById('videoResult');
                result.innerHTML = '⏳ Đang tạo video...';
                try {
                    const form = new FormData();
                    form.append('prompt', prompt);
                    const res = await fetch('/video', {method: 'POST', body: form});
                    const data = await res.json();
                    result.innerHTML = data.response || 'Không có phản hồi';
                } catch(e) {
                    result.innerHTML = '❌ Lỗi: ' + e.message;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
