# CLAUDE.md — VoxCPM (fork) project guide

> Read this first. It tells you what this repo is, what this fork adds, and how to set it
> up and run it on **any** machine. Paths that differ per-machine are flagged explicitly.

## What this is

A fork of [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) — a tokenizer-free TTS
(text-to-speech) model with zero-shot voice cloning. Current version: **VoxCPM2 (2.0.3)**,
model is 2B params, 48 kHz output, ZH/EN + ~30 languages.

**What this fork adds over upstream (the only fork-specific files — no model/inference changes):**

| File | Purpose |
|------|---------|
| `api_server.py` (+ `API_DOC.md`, `requirements-api.txt`) | A FastAPI **REST API** server. Upstream has no REST API — only the Python lib, CLI, and a Gradio demo. |
| `start.sh` (+ `MAC_STARTUP.md`) | **One-click Mac/Apple-Silicon (MPS) launcher** for the Gradio web UI, incl. the ffmpeg library-path workaround. |

Everything else (`app.py`, `src/voxcpm/`, `scripts/`, `conf/`, `docs/`) is upstream.

## Key files

- `src/voxcpm/` — the `voxcpm` package (installed editable). `core.py` = `VoxCPM` class (`from_pretrained`, `generate`).
- `app.py` — upstream Gradio web UI. Has `--model-id <path-or-hf-id>` and `--port` (default **8808**).
- `api_server.py` — fork's REST API. Config via env vars (see `API_DOC.md`). Default port 8000.
- `start.sh` — fork's Mac launcher: activates venv, sets ffmpeg path, runs `app.py` on port 7860.
- `models/VoxCPM2/` — model weights. **Gitignored — must be downloaded on each machine** (see Setup).
- `MAC_STARTUP.md` — Mac quick reference. `API_DOC.md` — full REST API docs (Swagger also at `/docs`).

## Setup on a new machine

```bash
# 1. Create venv + install (editable, so repo src/ is authoritative)
uv venv voxcpm-env --python 3.10        # or: python3.10 -m venv voxcpm-env
source voxcpm-env/bin/activate
uv pip install -e .                       # or: pip install -e .
uv pip install torchcodec                 # audio I/O

# 2. Download the model. Prefer ModelScope (faster on CN networks):
python -c "from modelscope import snapshot_download; snapshot_download('OpenBMB/VoxCPM2', local_dir='models/VoxCPM2')"
#   …or from HuggingFace: snapshot_download('openbmb/VoxCPM2') via huggingface_hub

# 3. (API only) install API deps
uv pip install -r requirements-api.txt
```

## Run

```bash
# Web UI (Mac): ./start.sh   ->  http://localhost:7860
# Web UI (manual / any OS):
python app.py --model-id models/VoxCPM2 --port 7860

# REST API:
python api_server.py         ->  http://localhost:8000/docs   (uses models/VoxCPM2 by default)

# CLI (single synthesis):
voxcpm --text "你好世界" --output out.wav --model-path models/VoxCPM2
```

## ⚠️ Machine-specific — adjust these per machine

1. **ffmpeg library path (macOS).** torchcodec needs ffmpeg libs on `DYLD_LIBRARY_PATH`.
   `start.sh` sets this automatically via `$(brew --prefix ffmpeg)/lib` (version-independent),
   so it works on any Mac with `brew install ffmpeg`. For manual commands use the same form:
   ```bash
   DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg)/lib" python app.py --model-id models/VoxCPM2 --port 7860
   ```
   On **Linux/Windows this is not needed** — drop `DYLD_LIBRARY_PATH`.

2. **Compute device** — auto-detected (CUDA → MPS → CPU), no flag needed.
   - Apple Silicon → MPS, and VoxCPM2 auto-uses **float32** (bf16 drifts on MPS).
   - NVIDIA box → CUDA + bf16, and `torch.compile` kicks in (much faster).

3. **Ports** — web UI 7860 (set in `start.sh`; app.py's own default is 8808), API 8000.

## REST API endpoints (`api_server.py`)

| Method | Path | Use |
|--------|------|-----|
| GET | `/health` | service + model status |
| GET | `/info` | model_id, sample_rate, device, dtype |
| POST | `/tts` | text → speech (JSON body `{"text": "..."}`) |
| POST | `/tts/clone` | voice cloning (multipart: `text`, `prompt_text`, `prompt_audio` file) |
| GET | `/docs` | Swagger UI |

## Notes

- `git` remotes: `origin` = this fork, `upstream` = OpenBMB/VoxCPM. Upgrade = `git merge upstream/main`.
- Measured RTF on Apple M4 Max / MPS: **VoxCPM2 ≈ 0.80** (float32). On an RTX 4090 upstream reports ~0.30.
- Don't commit `models/`, `voxcpm-env/` (gitignored).
