"""
FastAPI WebSocket server.

Protocol (client → server):
  - Binary frames:   Raw PCM Int16 LE audio at 16kHz
  - JSON text frame: {"type": "ping"} | {"type": "reset"}

Protocol (server → client):
  - {"type": "status",     "state": "listening|processing|speaking|idle"}
  - {"type": "transcript", "text": "...", "is_final": true|false}
  - {"type": "response",   "text": "..."}    # streamed LLM tokens
  - {"type": "latency",    "data": {...}}
  - {"type": "error",      "message": "..."}
  - Binary frames: WAV audio chunks for playback
"""

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from pipeline.orchestrator import ConversationSession, PipelineCallbacks, State
from config import settings


app = FastAPI(title="Jarvina — JarvisLabs Voice Assistant")

# Serve frontend from /frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info(f"New session: {ws.client}")

    async def on_state(state: State):
        await ws.send_text(json.dumps({"type": "status", "state": state.value}))

    async def on_transcript(text: str, is_final: bool):
        await ws.send_text(json.dumps({
            "type": "transcript",
            "text": text,
            "is_final": is_final,
        }))

    async def on_response_text(token: str):
        await ws.send_text(json.dumps({"type": "response", "text": token}))

    async def on_audio_chunk(audio_bytes: bytes):
        await ws.send_bytes(audio_bytes)

    async def on_latency(data: dict):
        await ws.send_text(json.dumps({"type": "latency", "data": data}))

    callbacks = PipelineCallbacks(
        on_state=on_state,
        on_transcript=on_transcript,
        on_response_text=on_response_text,
        on_audio_chunk=on_audio_chunk,
        on_latency=on_latency,
    )

    session = ConversationSession(callbacks)

    try:
        while True:
            msg = await ws.receive()

            if "bytes" in msg and msg["bytes"]:
                await session.feed_audio(msg["bytes"])

            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "reset":
                        session.vad.reset()
                        session.history.clear()
                        await on_state(State.LISTENING)
                        logger.info("Session reset")
                    elif data.get("type") == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"Session disconnected: {ws.client}")
    except Exception as exc:
        logger.exception(f"Session error: {exc}")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
        reload=False,
    )
