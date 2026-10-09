import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.vectorstore import build_index
from app.rag import answer_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()          # rebuild the index from transcripts once at startup
    yield


app = FastAPI(title="Rewind AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # fine for a student project; restrict after deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


def sse(event: str, data) -> str:
    # json.dumps keeps newlines inside tokens from breaking the SSE format
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history]

    async def event_generator():
        async for ev in answer_stream(req.question, history):
            yield sse(ev["event"], ev["data"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")