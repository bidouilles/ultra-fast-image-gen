# Ultra Fast Image Gen — HTTP API

FastAPI wrapper around the same diffusion pipelines used by the Gradio app
(`app.py`). Serves text-to-image generation over HTTP/JSON.

This document is written so it can be used as a reference for both humans and
agents (e.g. Claude Code) calling the API.

---

## TL;DR for agents

- **Base URL:** `http://<host>:7861` (e.g. `http://merlin.local:7861` on the LAN,
  `http://localhost:7861` on the host machine).
- **Generate an image:** `POST /generate` with JSON `{"prompt": "..."}`. Response
  body is raw PNG bytes. Add `"return_base64": true` to get JSON instead.
- **Default model:** `FLUX.2-klein-4B (4bit SDNQ - Low VRAM)`. Preloaded at
  startup. You don't need to call `/load` first.
- **Sensible defaults:** 512×512, 4 steps, CFG 3.5, random seed, first
  available device (MPS on Apple Silicon, CUDA on NVIDIA, else CPU).
- **Concurrency:** one GPU/MPS device serializes work. Don't fire many
  parallel `/generate` calls — they'll queue and may OOM. One request at a
  time is the safe default.
- **Switching models is expensive:** unloads the current pipeline and loads
  the new one (seconds → minutes, plus disk download on first use). Send the
  same `model` value across requests when possible.
- **Interactive docs:** open `/docs` in a browser for a Swagger UI you can
  click through.

---

## Running the server

```bash
# install deps (first time, or after pulling)
uv sync

# start the API (binds 0.0.0.0:7861 by default)
uv run python api.py

# override host/port
API_HOST=127.0.0.1 API_PORT=8080 uv run python api.py
```

The Gradio UI (`app.py`) uses port **7860**; the API uses **7861** so they can
run side-by-side.

The default model preloads during FastAPI's lifespan startup — the server
won't accept connections until the pipeline is in memory. First-ever startup
also downloads the model (~4–8 GB, cached under `~/.cache/huggingface`).

---

## Endpoints

### `GET /`
Service info. Lists available models and devices.

```json
{
  "name": "ultra-fast-image-gen",
  "default_model": "FLUX.2-klein-4B (4bit SDNQ - Low VRAM)",
  "available_models": ["FLUX.2-klein-4B (4bit SDNQ - Low VRAM)", "..."],
  "available_devices": ["mps", "cpu"]
}
```

### `GET /health`
Returns the currently-loaded model and device. `null` until the first load
completes.

```json
{ "status": "ok", "current_model": "flux2-klein-sdnq", "current_device": "mps" }
```

Note: `current_model` is the **internal** id (e.g. `flux2-klein-sdnq`), not the
display name passed in requests. See the model table below.

### `GET /models`
Lists the model display names accepted by `/load` and `/generate`.

### `POST /load`
Preload a model. Useful to warm the pipeline before a request you care about
the latency of, or to switch models out of band.

Query params:
- `model` (optional) — display name; defaults to the default model
- `device` (optional) — `mps` / `cuda` / `cpu`; defaults to first available

```bash
curl -X POST "http://localhost:7861/load?model=Z-Image%20Turbo%20(Quantized%20-%20Fast)"
```

### `POST /generate`
Generate an image from a prompt.

Request body (JSON):

| Field             | Type    | Default                                    | Range / notes                                            |
|-------------------|---------|--------------------------------------------|----------------------------------------------------------|
| `prompt`          | string  | **required**                               | The text prompt.                                         |
| `model`           | string? | default model                              | Must be one of `/models`. See table below.               |
| `width`           | int     | `512`                                      | 256–2048, multiples of 64 work best.                     |
| `height`          | int     | `512`                                      | 256–2048.                                                |
| `steps`           | int     | `4`                                        | 1–50. Klein/Z-Image Turbo are tuned for ~4 steps.        |
| `seed`            | int     | `-1`                                       | `-1` = random; otherwise deterministic.                  |
| `guidance_scale`  | float   | `3.5`                                      | 0–10. **FLUX:** 3.5 recommended. **Z-Image:** use `0`.   |
| `device`          | string? | first available                            | `mps` / `cuda` / `cpu`.                                  |
| `return_base64`   | bool    | `false`                                    | If true, returns JSON with base64 PNG instead of bytes.  |

**Default response** (`return_base64: false`):
- Status `200`
- `Content-Type: image/png`
- Body: raw PNG bytes
- Header `X-Generation-Info`: human-readable string with seed, model, mode, device, CFG

**JSON response** (`return_base64: true`):

```json
{
  "info": "Seed: 1234567 | Model: FLUX.2-klein-4B (4bit) | Mode: txt2img | Device: mps | CFG: 3.5",
  "image_base64": "iVBORw0KGgo...",
  "format": "png"
}
```

**Errors:**
- `400` — unknown model or unavailable device
- `422` — request body fails validation (e.g. width out of range)
- `500` — generation failure (often OOM); check server stdout

---

## Model choices

Pass these strings as the `model` field. The default is the first one.

| Display name (use this in requests)                  | Internal id            | VRAM (1024px) | Notes                              |
|------------------------------------------------------|------------------------|---------------|------------------------------------|
| `FLUX.2-klein-4B (4bit SDNQ - Low VRAM)` *(default)* | `flux2-klein-sdnq`     | ~16 GB        | Best balance. Img2img capable.\*   |
| `FLUX.2-klein-9B (4bit SDNQ - Higher Quality)`       | `flux2-klein-9b-sdnq`  | ~20 GB        | Higher quality, slower.            |
| `FLUX.2-klein-4B (Int8)`                             | `flux2-klein-int8`     | ~16 GB        | Older quantization scheme.         |
| `Z-Image Turbo (Quantized - Fast)`                   | `zimage-quant`         | ~8 GB         | Fastest. Text-to-image only.       |
| `Z-Image Turbo (Full - LoRA support)`                | `zimage-full`          | ~24 GB        | Supports LoRA (UI-only).           |

\* The HTTP API currently only exposes text-to-image. Image-to-image and LoRA
are wired up in `app.py` but not in `api.py` — extend `GenerateRequest` if you
need them.

---

## Examples

### curl — save PNG to disk

```bash
curl -X POST http://localhost:7861/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat astronaut on mars, cinematic, 35mm"}' \
  --output out.png
```

### curl — pull metadata from header

```bash
curl -X POST http://localhost:7861/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"...", "seed": 42}' \
  -D headers.txt --output out.png
grep -i "x-generation-info" headers.txt
```

### curl — JSON / base64 response

```bash
curl -X POST http://localhost:7861/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"...", "return_base64": true}' | jq .info
```

### Python — `requests`

```python
import requests

r = requests.post(
    "http://localhost:7861/generate",
    json={
        "prompt": "watercolor of a foggy harbor at dawn",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance_scale": 3.5,
        "seed": 12345,
    },
    timeout=600,  # generation can take a while on first model load
)
r.raise_for_status()
print(r.headers.get("X-Generation-Info"))
with open("out.png", "wb") as f:
    f.write(r.content)
```

### Python — switch model first, then generate

```python
import requests

base = "http://localhost:7861"

# Warm the pipeline (returns once the model is on-device)
requests.post(base + "/load", params={"model": "Z-Image Turbo (Quantized - Fast)"}).raise_for_status()

# Z-Image likes CFG=0
r = requests.post(base + "/generate", json={
    "prompt": "minimalist line drawing of a fox",
    "model": "Z-Image Turbo (Quantized - Fast)",
    "guidance_scale": 0,
    "steps": 4,
})
open("fox.png", "wb").write(r.content)
```

### Python — base64 inline (no file I/O)

```python
import base64, requests
r = requests.post("http://localhost:7861/generate", json={
    "prompt": "isometric pixel-art village",
    "return_base64": True,
}).json()
png_bytes = base64.b64decode(r["image_base64"])
```

---

## Tips for prompting / parameters

- **Klein (FLUX) → CFG 3.5**, 4 steps, 1024×1024 is the calibrated sweet
  spot. Going above ~6 steps rarely helps and costs proportional time.
- **Z-Image Turbo → CFG 0**, 4 steps. Setting CFG > 0 on Z-Image Turbo
  produces washed-out results — it's a CFG-distilled model.
- **Seeds** are reproducible per (model, device, steps, CFG, size, prompt).
  Same seed across different models will *not* give the same image.
- **Aspect ratio**: prefer multiples of 64 for both dimensions.
- **OOM at high resolution**: drop to 512×512, or switch to the 4-bit SDNQ
  model. The pipelines have attention slicing + VAE tiling enabled but very
  large outputs still blow past unified memory.

---

## Known limitations

- Text-to-image only via API — img2img / LoRA exist in the Gradio app only.
- No streaming / progress events. The HTTP request blocks until the image
  is ready.
- No auth. Bind to `127.0.0.1` (set `API_HOST=127.0.0.1`) if you don't want
  LAN exposure.
- Single-pipeline server. Switching models evicts the previous one from
  memory; concurrent users requesting different models will thrash.

---

## Source map

- `api.py` — FastAPI app, request schema, lifespan preload
- `app.py` — pipeline loaders (`load_*_pipeline`) and `generate_image`
  reused by the API
- `quantized_flux2.py` — int8 transformer wrapper for the FLUX int8 model
