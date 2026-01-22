
import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List

# Importar lógica do pipeline
from src.pipeline import executar_pipeline
from src.config import OUTPUT_DIR, VIDEO_SAIDA_BASE
from src.services.youtube import baixar_video_youtube, validar_url_youtube

app = FastAPI()

# Configurar CORS (Permitir frontend rodando em outra porta)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em prod, restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gerenciador de Conexões WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Diretórios
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Apenas eco ou keepalive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, "video_entrada.mp4")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": file_path}

@app.post("/download-youtube")
async def download_youtube(url: str = Form(...)):
    """
    Endpoint para baixar vídeo do YouTube.
    
    Args:
        url: URL do vídeo do YouTube
        
    Returns:
        JSON com status e caminho do arquivo
    """
    # Validar URL
    if not validar_url_youtube(url):
        return {"status": "error", "message": "URL do YouTube inválida"}
    
    file_path = os.path.join(UPLOAD_DIR, "video_entrada.mp4")
    
    # Remover arquivo anterior se existir
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass
    
    # Obter o event loop atual
    loop = asyncio.get_event_loop()
    
    # Callback para enviar progresso via WebSocket
    def progress_callback(msg):
        print(f"[YOUTUBE] {msg}")
        asyncio.run_coroutine_threadsafe(manager.broadcast(msg), loop)
    
    # Função wrapper para rodar no executor
    def run_download():
        return baixar_video_youtube(url, file_path, log_callback=progress_callback)
    
    # Executa blocking code em outra thread
    success = await asyncio.to_thread(run_download)
    
    if success:
        return {"status": "success", "path": file_path, "message": "Vídeo baixado com sucesso!"}
    else:
        return {"status": "error", "message": "Falha ao baixar o vídeo do YouTube"}

@app.post("/process")
async def process_video(
    motor: str = Form(...), 
    encoding: str = Form(...)
):
    video_path = os.path.join(UPLOAD_DIR, "video_entrada.mp4")
    
    if not os.path.exists(video_path):
        return {"error": "Vídeo não encontrado via upload."}

    # Obter o event loop atual
    loop = asyncio.get_event_loop()
    
    # Callback seguro para enviar mensagens via WebSocket
    def progress_callback(msg):
        print(f"[LOG] {msg}")
        # Agendar broadcast no event loop de forma thread-safe
        asyncio.run_coroutine_threadsafe(manager.broadcast(msg), loop)
    
    # Função wrapper para rodar no executor
    def run_pipeline():
        return executar_pipeline(
            caminho_video=video_path,
            idioma_origem="eng_Latn",
            idioma_destino="por_Latn",
            idioma_voz="por",
            motor_tts=motor,
            modo_encoding=encoding,
            progress_callback=progress_callback
        )

    # Executa blocking code em outra thread
    success = await asyncio.to_thread(run_pipeline)
    
    if success:
        return {"status": "success", "video_url": f"/download/{motor}"}
    else:
        return {"status": "error"}

@app.get("/download/{motor}")
async def download_video(motor: str):
    filename = f"video_dublado_{motor}.mp4"
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4", filename=filename)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
