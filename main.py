from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="INTELIGENT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
async def root():
    return {"name": "INTELIGENT API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    return {"response": f"INTELIGENT đã nhận: {req.prompt}"}

@app.post("/image")
async def image(prompt: str = Form(...)):
    return {"response": f"Tạo ảnh với: {prompt}"}

@app.post("/video")
async def video(prompt: str = Form(...)):
    return {"response": f"Tạo video với: {prompt}"}

@app.get("/web", response_class=HTMLResponse)
async def web():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>INTELIGENT</title>
    <style>
        body { font-family: Arial; background: #0a0a0f; color: #fff; padding: 20px; }
        .container { max-width: 600px; margin: auto; background: #1a1a2e; padding: 30px; border-radius: 20px; }
        h1 { text-align: center; color: #a78bfa; }
        input, textarea { width: 100%; padding: 10px; margin: 10px 0; border-radius: 8px; border: none; }
        button { padding: 10px 30px; background: #a78bfa; border: none; border-radius: 8px; color: #fff; cursor: pointer; }
        .result { background: #0a0a0f; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .tab { display: inline-block; padding: 10px 20px; cursor: pointer; background: #2a2a4a; border-radius: 8px; margin: 5px; }
        .tab.active { background: #a78bfa; }
        .content { display: none; margin-top: 20px; }
        .content.active { display: block; }
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 INTELIGENT</h1>
    <div style="text-align:center;margin-bottom:20px;">
        <span class="tab active" onclick="switchTab('chat')">💬 Chat</span>
        <span class="tab" onclick="switchTab('image')">🎨 Ảnh</span>
        <span class="tab" onclick="switchTab('video')">🎬 Video</span>
    </div>

    <div id="chat" class="content active">
        <textarea id="chatInput" rows="3">Xin chào!</textarea>
        <button onclick="sendChat()">Gửi</button>
        <div id="chatResult" class="result">Nhập tin nhắn...</div>
    </div>

    <div id="image" class="content">
        <input id="imagePrompt" value="mountain sunset">
        <button onclick="genImage()">Tạo ảnh</button>
        <div id="imageResult" class="result">Nhập mô tả...</div>
    </div>

    <div id="video" class="content">
        <input id="videoPrompt" value="cat walking">
        <button onclick="genVideo()">Tạo video</button>
        <div id="videoResult" class="result">Nhập mô tả...</div>
    </div>
</div>

<script>
function switchTab(tab) {
    document.querySelectorAll('.content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById(tab).classList.add('active');
    event.target.classList.add('active');
}

async function sendChat() {
    const prompt = document.getElementById('chatInput').value;
    const result = document.getElementById('chatResult');
    result.innerHTML = '⏳...';
    const res = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt})
    });
    const data = await res.json();
    result.innerHTML = data.response;
}

async function genImage() {
    const prompt = document.getElementById('imagePrompt').value;
    const result = document.getElementById('imageResult');
    result.innerHTML = '⏳...';
    const form = new FormData();
    form.append('prompt', prompt);
    const res = await fetch('/image', {method: 'POST', body: form});
    const data = await res.json();
    result.innerHTML = data.response;
}

async function genVideo() {
    const prompt = document.getElementById('videoPrompt').value;
    const result = document.getElementById('videoResult');
    result.innerHTML = '⏳...';
    const form = new FormData();
    form.append('prompt', prompt);
    const res = await fetch('/video', {method: 'POST', body: form});
    const data = await res.json();
    result.innerHTML = data.response;
}
</script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
