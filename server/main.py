import io
import os
import time
import uuid
import wave
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.hf_dataset_writer import HuggingFaceDatasetWriter

MAX_DURATION_SECONDS = 30.0
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 20
ALLOWED_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}

_rate_limit_store: Dict[str, Deque[float]] = defaultdict(deque)


def _parse_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    if "*" in origins:
        return ["*"]
    return origins


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _enforce_rate_limit(client_ip: str) -> None:
    now = time.time()
    requests = _rate_limit_store[client_ip]
    while requests and (now - requests[0]) > RATE_LIMIT_WINDOW_SECONDS:
        requests.popleft()
    if len(requests) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry in 1 minute.")
    requests.append(now)


def _wav_duration_seconds(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                raise ValueError("Invalid WAV sample rate.")
            return frame_count / float(frame_rate)
    except wave.Error as error:
        raise HTTPException(status_code=400, detail=f"Invalid WAV audio: {error}") from error


app = FastAPI(title="Voice Dataset Uploader API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(),
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

hf_writer: HuggingFaceDatasetWriter | None = None


@app.on_event("startup")
def startup_event() -> None:
    global hf_writer
    token = _require_env("HF_TOKEN")
    dataset_repo = _require_env("HF_DATASET_REPO_ID")
    hf_writer = HuggingFaceDatasetWriter(token=token, repo_id=dataset_repo, private=False)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tools/voice-dataset/submit")
async def submit_voice_sample(
    request: Request,
    audio: UploadFile = File(...),
    transcript: str = Form(...),
    language: str = Form("en"),
) -> dict:
    if hf_writer is None:
        raise HTTPException(status_code=500, detail="Server is not initialized.")

    client_ip = request.client.host if request.client else "unknown"
    _enforce_rate_limit(client_ip)

    normalized_transcript = transcript.strip()
    if not normalized_transcript:
        raise HTTPException(status_code=400, detail="Transcript is required.")

    if audio.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Only WAV uploads are accepted.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large.")

    duration_sec = _wav_duration_seconds(audio_bytes)
    if duration_sec > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds {MAX_DURATION_SECONDS:.0f} seconds. Received {duration_sec:.2f} seconds.",
        )

    clean_language = (language or "en").strip() or "en"
    sample_id = uuid.uuid4().hex

    result = hf_writer.upload_sample(
        sample_id=sample_id,
        audio_bytes=audio_bytes,
        transcript=normalized_transcript,
        language=clean_language,
        duration_sec=duration_sec,
    )

    return {
        "ok": True,
        "dataset_repo": result.dataset_repo,
        "sample_id": result.sample_id,
        "audio_path": result.audio_path,
        "jsonl_path": result.jsonl_path,
        "duration_sec": round(duration_sec, 2),
        "commit_url": result.commit_url,
    }
