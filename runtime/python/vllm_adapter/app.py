import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from cosyvoice.utils.file_utils import load_wav
from vllm import ModelRegistry

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM


ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

BASE_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    model_dir = os.getenv("COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B")
    api_token = os.getenv("COSYVOICE_API_TOKEN", "").strip()
    load_vllm = _env_flag("COSYVOICE_LOAD_VLLM", True)
    load_trt = _env_flag("COSYVOICE_LOAD_TRT", False)
    fp16 = _env_flag("COSYVOICE_FP16", False)
    default_speed = float(os.getenv("COSYVOICE_DEFAULT_SPEED", "1.0"))
    default_voice = os.getenv("COSYVOICE_DEFAULT_VOICE", "")
    warmup_text = os.getenv("COSYVOICE_WARMUP_TEXT", "你好，欢迎使用 CosyVoice。")
    profiles_path = Path(os.getenv("COSYVOICE_SPEAKERS_JSON", str(BASE_DIR / "speakers.example.json")))


SETTINGS = Settings()
MODEL = None
MODEL_LOCK = threading.Lock()
VOICE_PROFILES: dict[str, dict[str, str]] = {}


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    model: str | None = None
    sample_rate: int | None = None
    stream: bool = True
    speed: float | None = None


def _require_token(authorization: str | None = Header(default=None)) -> None:
    expected = SETTINGS.api_token
    if not expected:
        return
    actual = (authorization or "").strip()
    if actual != f"Bearer {expected}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token")


def _load_profiles() -> dict[str, dict[str, str]]:
    path = SETTINGS.profiles_path
    if not path.exists():
        raise RuntimeError(f"speaker profiles file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("voices") if isinstance(data, dict) else data
    if not isinstance(profiles, dict) or not profiles:
        raise RuntimeError("speaker profiles file is empty or invalid")

    normalized: dict[str, dict[str, str]] = {}
    for voice_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise RuntimeError(f"profile for {voice_name} must be an object")
        prompt_text = str(profile.get("prompt_text") or "").strip()
        prompt_wav = str(profile.get("prompt_wav") or "").strip()
        if not prompt_text or not prompt_wav:
            raise RuntimeError(f"profile for {voice_name} requires prompt_text and prompt_wav")
        wav_path = Path(prompt_wav)
        if not wav_path.is_absolute():
            wav_path = (path.parent / wav_path).resolve()
        if not wav_path.exists():
            raise RuntimeError(f"prompt_wav for {voice_name} not found: {wav_path}")
        normalized[str(voice_name)] = {
            "prompt_text": prompt_text,
            "prompt_wav": str(wav_path),
        }
    return normalized


def _resolve_voice(req: TTSRequest) -> tuple[str, dict[str, str]]:
    voice = (req.voice or SETTINGS.default_voice).strip()
    if not voice:
        raise HTTPException(status_code=400, detail="voice is required")
    profile = VOICE_PROFILES.get(voice)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"unknown voice: {voice}")
    return voice, profile


def _get_model():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return MODEL


def _pcm_stream(req: TTSRequest) -> Iterator[bytes]:
    cosyvoice = _get_model()
    requested_sample_rate = req.sample_rate or cosyvoice.sample_rate
    if requested_sample_rate != cosyvoice.sample_rate:
        raise HTTPException(
            status_code=400,
            detail=f"sample_rate {requested_sample_rate} is not supported, expected {cosyvoice.sample_rate}",
        )

    _, profile = _resolve_voice(req)
    speed = req.speed or SETTINGS.default_speed
    prompt_wav = load_wav(profile["prompt_wav"], 16000)

    with MODEL_LOCK:
        for item in cosyvoice.inference_zero_shot(
            req.text,
            profile["prompt_text"],
            prompt_wav,
            stream=req.stream,
            speed=speed,
        ):
            audio = item["tts_speech"].detach().cpu().numpy()
            pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
            yield pcm.tobytes()


def _load_model() -> None:
    global MODEL, VOICE_PROFILES
    VOICE_PROFILES = _load_profiles()
    MODEL = AutoModel(
        model_dir=SETTINGS.model_dir,
        load_vllm=SETTINGS.load_vllm,
        load_trt=SETTINGS.load_trt,
        fp16=SETTINGS.fp16,
    )


def _run_warmup() -> None:
    if not SETTINGS.warmup_text or not VOICE_PROFILES:
        return
    _, profile = next(iter(VOICE_PROFILES.items()))
    prompt_wav = load_wav(profile["prompt_wav"], 16000)
    for _ in MODEL.inference_zero_shot(
        SETTINGS.warmup_text,
        profile["prompt_text"],
        prompt_wav,
        stream=True,
        speed=SETTINGS.default_speed,
    ):
        break


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    _run_warmup()
    yield


app = FastAPI(title="CosyVoice3 vLLM Zero-Shot Adapter", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    cosyvoice = _get_model()
    return {
        "ok": True,
        "model_dir": SETTINGS.model_dir,
        "sample_rate": cosyvoice.sample_rate,
        "voices": sorted(VOICE_PROFILES.keys()),
        "profiles_path": str(SETTINGS.profiles_path),
        "load_vllm": SETTINGS.load_vllm,
        "load_trt": SETTINGS.load_trt,
    }


@app.get("/v1/voices", dependencies=[Depends(_require_token)])
async def list_voices():
    return {"data": [{"voice": name} for name in sorted(VOICE_PROFILES.keys())]}


@app.post("/v1/tts", dependencies=[Depends(_require_token)])
async def tts(req: TTSRequest):
    return StreamingResponse(_pcm_stream(req), media_type="audio/pcm")


@app.post("/v1/tts/pcm16", dependencies=[Depends(_require_token)])
async def tts_pcm16(req: TTSRequest):
    return StreamingResponse(_pcm_stream(req), media_type="audio/pcm")


@app.get("/")
async def root():
    return JSONResponse(
        {
            "service": "cosyvoice3-vllm-zeroshot-adapter",
            "routes": ["/healthz", "/readyz", "/v1/voices", "/v1/tts", "/v1/tts/pcm16"],
        }
    )
