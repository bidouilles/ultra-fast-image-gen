"""
FastAPI interface for ultra-fast image generation.

Reuses pipeline loading from app.py. Defaults to FLUX.2-klein-4B (4bit SDNQ - Low VRAM).

Run:
    uv run python api.py
    # or override host/port:
    API_HOST=0.0.0.0 API_PORT=7861 uv run python api.py

Endpoints:
    GET  /              - service info
    GET  /health        - status, current loaded model/device
    GET  /models        - list available model choices
    POST /load          - preload a model: ?model=...&device=...
    POST /generate      - generate image, returns PNG bytes
"""

import io
import os
import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

import app as _app


DEFAULT_MODEL = "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)"


def _default_device() -> str:
    devices = _app.get_available_devices()
    return devices[0] if devices else "cpu"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    model: Optional[str] = Field(default=None, description=f"Model choice (default: {DEFAULT_MODEL!r})")
    width: int = Field(default=512, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    steps: int = Field(default=4, ge=1, le=50)
    seed: int = Field(default=-1, description="-1 for random")
    guidance_scale: float = Field(default=3.5, ge=0.0, le=10.0)
    device: Optional[str] = Field(default=None, description="mps/cuda/cpu (default: first available)")
    return_base64: bool = Field(default=False, description="If true, return JSON with base64-encoded PNG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    dev = _default_device()
    print(f"[startup] Preloading default model: {DEFAULT_MODEL} on {dev}")
    try:
        _app.load_pipeline(DEFAULT_MODEL, dev)
        print(f"[startup] Default model ready.")
    except Exception as e:
        print(f"[startup] Warning — could not preload default model: {e}")
    yield


api = FastAPI(
    title="Ultra Fast Image Gen API",
    version="0.1.0",
    description="FastAPI wrapper around the diffusers pipelines used by app.py.",
    lifespan=lifespan,
)


@api.get("/")
def root():
    return {
        "name": "ultra-fast-image-gen",
        "default_model": DEFAULT_MODEL,
        "available_models": _app.MODEL_CHOICES,
        "available_devices": _app.get_available_devices(),
    }


@api.get("/health")
def health():
    return {
        "status": "ok",
        "current_model": _app.current_model,
        "current_device": _app.current_device,
    }


@api.get("/models")
def models():
    return {"models": _app.MODEL_CHOICES, "default": DEFAULT_MODEL}


@api.post("/load")
def load(model: Optional[str] = None, device: Optional[str] = None):
    model_choice = model or DEFAULT_MODEL
    if model_choice not in _app.MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model. Choose from: {_app.MODEL_CHOICES}")
    dev = device or _default_device()
    if dev not in _app.get_available_devices():
        raise HTTPException(400, f"Device {dev} unavailable. Available: {_app.get_available_devices()}")
    _app.load_pipeline(model_choice, dev)
    return {"loaded_model": _app.current_model, "device": _app.current_device}


@api.post("/generate")
def generate(req: GenerateRequest):
    model_choice = req.model or DEFAULT_MODEL
    if model_choice not in _app.MODEL_CHOICES:
        raise HTTPException(400, f"Unknown model. Choose from: {_app.MODEL_CHOICES}")
    dev = req.device or _default_device()
    if dev not in _app.get_available_devices():
        raise HTTPException(400, f"Device {dev} unavailable. Available: {_app.get_available_devices()}")

    image, info = _app.generate_image(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        steps=req.steps,
        seed=req.seed,
        guidance=req.guidance_scale,
        device=dev,
        model_choice=model_choice,
        input_images=None,
        lora_file=None,
        lora_strength=1.0,
        auto_save=False,
        output_dir=_app.DEFAULT_OUTPUT_DIR,
    )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    if req.return_base64:
        return {
            "info": info,
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            "format": "png",
        }

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"X-Generation-Info": info.replace("\n", " ")},
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "7861"))
    uvicorn.run(api, host=host, port=port)
