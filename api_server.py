#!/usr/bin/env python3
"""
VoxCPM TTS API Server

A robust FastAPI-based HTTP API for text-to-speech and voice cloning.

Usage:
    python api_server.py [--host HOST] [--port PORT]

Environment Variables:
    API_HOST                  - Server host (default: 0.0.0.0)
    API_PORT                  - Server port (default: 8000)
    HF_MODEL_ID               - HuggingFace model ID (default: openbmb/VoxCPM1.5)
    MODEL_PATH                - Local model path (overrides HF_MODEL_ID)
    LOAD_DENOISER             - Load denoiser model (default: true)
    MAX_CONCURRENT_REQUESTS   - Max concurrent inference requests (default: 1)
    REQUEST_TIMEOUT_SECONDS   - Request timeout in seconds (default: 300)
    MAX_UPLOAD_SIZE_MB        - Max upload file size in MB (default: 50)
    DYLD_LIBRARY_PATH         - FFmpeg library path (required on macOS)
"""

import argparse
import asyncio
import io
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

import voxcpm
from voxcpm.model.voxcpm import LoRAConfig


# =============================================================================
# Configuration
# =============================================================================

class Settings(BaseSettings):
    """API server configuration from environment variables."""

    # Server
    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")

    # Model
    hf_model_id: str = Field(default="openbmb/VoxCPM1.5", alias="HF_MODEL_ID")
    model_path: Optional[str] = Field(default=None, alias="MODEL_PATH")
    cache_dir: Optional[str] = Field(default=None, alias="CACHE_DIR")
    local_files_only: bool = Field(default=False, alias="LOCAL_FILES_ONLY")
    load_denoiser: bool = Field(default=True, alias="LOAD_DENOISER")
    zipenhancer_model_id: str = Field(
        default="iic/speech_zipenhancer_ans_multiloss_16k_base",
        alias="ZIPENHANCER_MODEL_ID"
    )
    optimize: bool = Field(default=True, alias="OPTIMIZE_MODEL")

    # LoRA
    lora_weights_path: Optional[str] = Field(default=None, alias="LORA_WEIGHTS_PATH")
    lora_enable_lm: bool = Field(default=True, alias="LORA_ENABLE_LM")
    lora_enable_dit: bool = Field(default=True, alias="LORA_ENABLE_DIT")
    lora_r: int = Field(default=32, alias="LORA_R")
    lora_alpha: int = Field(default=16, alias="LORA_ALPHA")

    # Concurrency & Timeout
    max_concurrent_requests: int = Field(default=1, alias="MAX_CONCURRENT_REQUESTS")
    request_timeout_seconds: int = Field(default=300, alias="REQUEST_TIMEOUT_SECONDS")

    # File Upload
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    allowed_audio_extensions: list[str] = Field(
        default=[".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    )

    # Temp Files
    temp_dir: Optional[str] = Field(default=None, alias="TEMP_DIR")

    class Config:
        env_file = ".env"
        populate_by_name = True


settings = Settings()


# =============================================================================
# Pydantic Models (Request/Response)
# =============================================================================

class TTSRequest(BaseModel):
    """TTS request parameters."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    cfg_value: float = Field(default=2.0, ge=1.0, le=5.0, description="CFG guidance value")
    inference_timesteps: int = Field(default=10, ge=4, le=50, description="Inference steps")
    min_len: int = Field(default=2, ge=1, description="Minimum output length")
    max_len: int = Field(default=4096, ge=10, le=8192, description="Maximum output length")
    normalize: bool = Field(default=False, description="Enable text normalization")
    denoise: bool = Field(default=False, description="Denoise reference audio")
    retry_badcase: bool = Field(default=True, description="Retry on bad cases")
    retry_badcase_max_times: int = Field(default=3, ge=1, le=10, description="Max retry times")
    retry_badcase_ratio_threshold: float = Field(default=6.0, ge=1.0, le=20.0)
    output_format: Literal["wav", "mp3", "flac"] = Field(default="wav")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        return v


class TTSCloneRequest(TTSRequest):
    """Voice clone request parameters."""

    prompt_text: str = Field(..., min_length=1, max_length=2000, description="Reference audio text")

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Prompt text cannot be empty")
        return v


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy", "loading"]
    model_loaded: bool
    device: str
    message: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_id: str
    sample_rate: int
    device: str
    dtype: str
    lora_enabled: bool
    denoiser_available: bool


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
    error_code: str


# =============================================================================
# Model Manager (Singleton)
# =============================================================================

class ModelManager:
    """Thread-safe singleton model manager."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model: Optional[voxcpm.VoxCPM] = None
        self._loading = False
        self._load_error: Optional[str] = None
        self._initialized = True

    @property
    def model(self) -> Optional[voxcpm.VoxCPM]:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_model(self) -> voxcpm.VoxCPM:
        """Load model (thread-safe)."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            self._loading = True
            self._load_error = None

            try:
                print(f"Loading model: {settings.model_path or settings.hf_model_id}")

                # Build LoRA config if weights path provided
                lora_config = None
                if settings.lora_weights_path:
                    lora_config = LoRAConfig(
                        enable_lm=settings.lora_enable_lm,
                        enable_dit=settings.lora_enable_dit,
                        r=settings.lora_r,
                        alpha=settings.lora_alpha,
                    )

                # Load model
                if settings.model_path:
                    self._model = voxcpm.VoxCPM(
                        voxcpm_model_path=settings.model_path,
                        zipenhancer_model_path=(
                            settings.zipenhancer_model_id if settings.load_denoiser else None
                        ),
                        enable_denoiser=settings.load_denoiser,
                        optimize=settings.optimize,
                        lora_config=lora_config,
                        lora_weights_path=settings.lora_weights_path,
                    )
                else:
                    self._model = voxcpm.VoxCPM.from_pretrained(
                        hf_model_id=settings.hf_model_id,
                        load_denoiser=settings.load_denoiser,
                        zipenhancer_model_id=settings.zipenhancer_model_id,
                        cache_dir=settings.cache_dir,
                        local_files_only=settings.local_files_only,
                        optimize=settings.optimize,
                        lora_config=lora_config,
                        lora_weights_path=settings.lora_weights_path,
                    )

                print(f"Model loaded! Sample rate: {self._model.tts_model.sample_rate}")
                return self._model

            except Exception as e:
                self._load_error = str(e)
                raise RuntimeError(f"Failed to load model: {e}")
            finally:
                self._loading = False


model_manager = ModelManager()


# =============================================================================
# Temp File Manager
# =============================================================================

class TempFileManager:
    """Temporary file manager with automatic cleanup."""

    def __init__(self):
        self.temp_dir = settings.temp_dir or tempfile.gettempdir()
        os.makedirs(self.temp_dir, exist_ok=True)

    def create_temp_path(self, suffix: str = ".wav") -> str:
        """Create a unique temp file path."""
        filename = f"voxcpm_{uuid.uuid4().hex}{suffix}"
        return os.path.join(self.temp_dir, filename)

    @staticmethod
    def safe_delete(file_path: Optional[str]) -> bool:
        """Safely delete a file."""
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                return True
            except OSError:
                return False
        return False


temp_manager = TempFileManager()


# =============================================================================
# Inference Executor
# =============================================================================

_executor = ThreadPoolExecutor(max_workers=2)
_inference_semaphore: Optional[asyncio.Semaphore] = None


def get_inference_semaphore() -> asyncio.Semaphore:
    """Get or create inference semaphore."""
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    return _inference_semaphore


def get_model() -> voxcpm.VoxCPM:
    """Dependency: get loaded model."""
    if not model_manager.is_loaded:
        if model_manager.load_error:
            raise HTTPException(503, f"Model failed to load: {model_manager.load_error}")
        raise HTTPException(503, "Model not loaded")
    return model_manager.model


# =============================================================================
# TTS Service
# =============================================================================

def generate_audio_sync(
    model: voxcpm.VoxCPM,
    text: str,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    min_len: int = 2,
    max_len: int = 4096,
    normalize: bool = False,
    denoise: bool = False,
    retry_badcase: bool = True,
    retry_badcase_max_times: int = 3,
    retry_badcase_ratio_threshold: float = 6.0,
) -> np.ndarray:
    """Synchronous audio generation."""
    return model.generate(
        text=text,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        min_len=min_len,
        max_len=max_len,
        normalize=normalize,
        denoise=denoise,
        retry_badcase=retry_badcase,
        retry_badcase_max_times=retry_badcase_max_times,
        retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
    )


def array_to_bytes(audio: np.ndarray, sample_rate: int, fmt: str = "wav") -> bytes:
    """Convert numpy array to audio bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=fmt)
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    print("Starting VoxCPM API server...")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, model_manager.load_model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load model at startup: {e}")

    yield

    print("Shutting down VoxCPM API server...")


app = FastAPI(
    title="VoxCPM TTS API",
    description="""
VoxCPM Text-to-Speech API Service

## Features
- **Text-to-Speech (TTS)**: Convert text to high-quality speech
- **Voice Cloning**: Clone voice using reference audio

## Endpoints
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /tts` - Text to speech
- `POST /tts/clone` - Voice cloning
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc), "error_code": "INTERNAL_ERROR"},
    )


# Timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(
            call_next(request),
            timeout=settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Request Timeout",
                "detail": f"Request exceeded {settings.request_timeout_seconds}s",
                "error_code": "TIMEOUT",
            },
        )


# =============================================================================
# Routes
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check service health status."""
    if model_manager.is_loading:
        return HealthResponse(
            status="loading", model_loaded=False, device="unknown",
            message="Model is loading...",
        )

    if model_manager.load_error:
        return HealthResponse(
            status="unhealthy", model_loaded=False, device="unknown",
            message=f"Load failed: {model_manager.load_error}",
        )

    if not model_manager.is_loaded:
        return HealthResponse(
            status="unhealthy", model_loaded=False, device="unknown",
            message="Model not loaded",
        )

    model = model_manager.model
    return HealthResponse(
        status="healthy", model_loaded=True,
        device=str(model.tts_model.device),
        message="Service is running",
    )


@app.get("/info", response_model=ModelInfoResponse, tags=["Health"])
async def model_info(model: Annotated[voxcpm.VoxCPM, Depends(get_model)]):
    """Get model information."""
    return ModelInfoResponse(
        model_id=settings.model_path or settings.hf_model_id,
        sample_rate=model.tts_model.sample_rate,
        device=str(model.tts_model.device),
        dtype=model.tts_model.config.dtype,
        lora_enabled=model.lora_enabled,
        denoiser_available=model.denoiser is not None,
    )


@app.post("/tts", response_class=Response, tags=["TTS"])
async def text_to_speech(
    request: TTSRequest,
    model: Annotated[voxcpm.VoxCPM, Depends(get_model)],
):
    """
    Convert text to speech.

    Returns audio file in specified format (wav/mp3/flac).
    """
    semaphore = get_inference_semaphore()

    try:
        async with semaphore:
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                _executor,
                generate_audio_sync,
                model,
                request.text,
                None,  # prompt_wav_path
                None,  # prompt_text
                request.cfg_value,
                request.inference_timesteps,
                request.min_len,
                request.max_len,
                request.normalize,
                False,  # denoise
                request.retry_badcase,
                request.retry_badcase_max_times,
                request.retry_badcase_ratio_threshold,
            )

        sample_rate = model.tts_model.sample_rate
        duration = len(audio) / sample_rate
        audio_bytes = array_to_bytes(audio, sample_rate, request.output_format)

        media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}

        return Response(
            content=audio_bytes,
            media_type=media_types.get(request.output_format, "audio/wav"),
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Duration-Seconds": f"{duration:.2f}",
                "X-Text-Length": str(len(request.text)),
                "Content-Disposition": f'attachment; filename="output.{request.output_format}"',
            },
        )

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")


@app.post("/tts/clone", response_class=Response, tags=["TTS"])
async def voice_clone(
    background_tasks: BackgroundTasks,
    model: Annotated[voxcpm.VoxCPM, Depends(get_model)],
    # Form fields
    text: Annotated[str, Form(..., min_length=1, max_length=10000)],
    prompt_text: Annotated[str, Form(..., min_length=1, max_length=2000)],
    cfg_value: Annotated[float, Form(ge=1.0, le=5.0)] = 2.0,
    inference_timesteps: Annotated[int, Form(ge=4, le=50)] = 10,
    min_len: Annotated[int, Form(ge=1)] = 2,
    max_len: Annotated[int, Form(ge=10, le=8192)] = 4096,
    normalize: Annotated[bool, Form()] = False,
    denoise: Annotated[bool, Form()] = False,
    retry_badcase: Annotated[bool, Form()] = True,
    retry_badcase_max_times: Annotated[int, Form(ge=1, le=10)] = 3,
    retry_badcase_ratio_threshold: Annotated[float, Form(ge=1.0, le=20.0)] = 6.0,
    output_format: Annotated[str, Form()] = "wav",
    # File upload
    prompt_audio: UploadFile = File(..., description="Reference audio file"),
):
    """
    Clone voice using reference audio.

    Upload reference audio file via multipart/form-data.
    """
    # Validate file extension
    if prompt_audio.filename:
        ext = os.path.splitext(prompt_audio.filename)[1].lower()
        if ext not in settings.allowed_audio_extensions:
            raise HTTPException(400, f"Unsupported format: {ext}")

    # Save uploaded file
    temp_path = temp_manager.create_temp_path()

    try:
        content = await prompt_audio.read()

        # Check file size
        max_size = settings.max_upload_size_mb * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(400, f"File too large. Max: {settings.max_upload_size_mb}MB")

        with open(temp_path, "wb") as f:
            f.write(content)

        semaphore = get_inference_semaphore()

        async with semaphore:
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                _executor,
                generate_audio_sync,
                model,
                text.strip(),
                temp_path,
                prompt_text.strip(),
                cfg_value,
                inference_timesteps,
                min_len,
                max_len,
                normalize,
                denoise,
                retry_badcase,
                retry_badcase_max_times,
                retry_badcase_ratio_threshold,
            )

        # Schedule temp file cleanup
        background_tasks.add_task(temp_manager.safe_delete, temp_path)

        sample_rate = model.tts_model.sample_rate
        duration = len(audio) / sample_rate
        audio_bytes = array_to_bytes(audio, sample_rate, output_format)

        media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}

        return Response(
            content=audio_bytes,
            media_type=media_types.get(output_format, "audio/wav"),
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Duration-Seconds": f"{duration:.2f}",
                "X-Text-Length": str(len(text)),
                "Content-Disposition": f'attachment; filename="output.{output_format}"',
            },
        )

    except HTTPException:
        temp_manager.safe_delete(temp_path)
        raise
    except ValueError as e:
        temp_manager.safe_delete(temp_path)
        raise HTTPException(400, str(e))
    except Exception as e:
        temp_manager.safe_delete(temp_path)
        raise HTTPException(500, f"Generation failed: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VoxCPM TTS API Server")
    parser.add_argument("--host", type=str, default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"Starting VoxCPM API on http://{args.host}:{args.port}")
    print(f"Model: {settings.model_path or settings.hf_model_id}")
    print(f"Max concurrent: {settings.max_concurrent_requests}")
    print(f"Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
