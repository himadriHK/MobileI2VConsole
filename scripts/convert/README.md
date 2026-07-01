# MobileI2V — ONNX Model Conversion

Convert the four PyTorch models of the MobileI2V pipeline to ONNX for
Android mobile inference via [onnxruntime](https://onnxruntime.ai/).

## Pipeline

```
Image (720p)        Prompt ("a dog running")
    │                     │
    ▼                     ▼
VAE Encoder ──► Qwen2-0.5B Encoder
    │                     │
    └──────┬──────────────┘
           ▼
    MobileI2V UNet (2-step diffusion)
           │
           ▼
    Turbo-VAED Decoder
           │
           ▼
    17 RGB Frames  →  H.264 .mp4
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.6+ support (recommended; falls back to CPU)
- NVIDIA driver >= 560.76 (for CUDA 12.6 on Windows; check with `nvidia-smi`)

## Setup

```bash
pip install -r requirements.txt
```

### GPU Setup (CUDA 12.6)

```bash
# 1. Install PyTorch with CUDA 12.6 support
pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu126

# 2. Install ONNX Runtime GPU (replace CPU-only onnxruntime)
pip uninstall onnxruntime
pip install onnxruntime-gpu>=1.19.0

# 3. Install remaining dependencies
pip install -r requirements.txt
```

> ⚠️ `onnxruntime` (CPU) and `onnxruntime-gpu` (GPU) conflict — install one or the other, not both.

## Usage

### Convert all models at once

```bash
python convert_all.py --output ./models/
```

For mobile-optimized ORT format (requires `onnxruntime-tools`):

```bash
python convert_all.py --output ./models/ --optimize
```

### Convert individual models

Each model can be converted independently:

```bash
python convert_vae_encoder.py --output ./models/
python convert_qwen2_encoder.py --output ./models/
python convert_mobilei2v_unet.py --output ./models/
python convert_turbo_vaed.py --output ./models/
```

### Command-line options

All converters accept:

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output directory (default: `./models/`) |
| `--verbose`, `-v` | Print detailed export progress |
| `--device` | Override device (`cuda` / `cpu`) |

Additional options per model:

| Script | Option | Default | Description |
|--------|--------|---------|-------------|
| `convert_vae_encoder.py` | `--height` / `--width` | 720 / 1280 | Example image dimensions |
| `convert_qwen2_encoder.py` | `--max-seq-len` | 77 | Maximum token sequence length |
| `convert_mobilei2v_unet.py` | `--height` / `--width` | 90 / 160 | Latent spatial dimensions |
| `convert_mobilei2v_unet.py` | `--seq-len` | 77 | Text embedding sequence length |
| `convert_turbo_vaed.py` | `--height` / `--width` | 90 / 160 | Latent spatial dimensions |

## Output Files

After a successful conversion, the output directory contains:

| File | Source | Size (est.) |
|------|--------|-------------|
| `vae_encoder.onnx` | LTX-Video VAE (encoder) | ~50 MB |
| `qwen2_encoder.onnx` | Qwen/Qwen2-0.5B | ~900 MB |
| `mobilei2v_unet.onnx` | MobileI2V 270M UNet | ~540 MB |
| `turbo_vaed.onnx` | Mobile-optimized VAE decoder | ~50 MB |
| `models_manifest.json` | Conversion manifest | — |
| `*.ort` | (optional) ORT mobile format | varies |

### models_manifest.json

```json
{
  "generated_at": "2026-07-01T12:00:00Z",
  "models": {
    "vae_encoder": {
      "format": "onnx",
      "path": "models/vae_encoder.onnx",
      "size_bytes": 52428800,
      "sha256": "abc123..."
    },
    "qwen2_encoder": { ... },
    "mobilei2v_unet": { ... },
    "turbo_vaed": { ... }
  }
}
```

## Model Architectures

### VAE Encoder
- **Source:** `Lightricks/LTX-Video` (HuggingFace)
- **Input:** `[1, 3, H, W]` float32 RGB image
- **Output:** `[1, 4, H/8, W/8]` float32 latent
- **Spatial compression:** 8× (3× downsampling conv layers)

### Qwen2-0.5B Text Encoder
- **Source:** `Qwen/Qwen2-0.5B` (HuggingFace)
- **Input:** `[1, seq_len]` int64 token IDs + attention mask
- **Output:** `[1, seq_len, 1024]` float32 embeddings
- **Architecture:** 24-layer transformer, 1024 hidden dim

### MobileI2V Denoising UNet
- **Source:** `hustvl/MobileI2V` (HuggingFace)
- **Inputs:** latent `[1, 4, 9, H/8, W/8]`, text `[1, 77, 1024]`, timestep `[1]`
- **Output:** `[1, 4, 9, H/8, W/8]` float32 denoised latent
- **Parameters:** ~270M, hybrid spatial-temporal attention
- **Distillation:** 2-step (t=1.0 noise prediction, t=0.0 clean prediction)

### Turbo-VAED Decoder
- **Source:** Mobile-optimized VAE decoder (HuggingFace)
- **Input:** `[17, 4, H/8, W/8]` float32 latent frames
- **Output:** `[17, 3, H, W]` float32 RGB frames
- **Details:** Lightweight decoder with grouped convolutions

## Notes

- Models export with **ONNX opset 17**.
- On **CPU**: models use **float32** precision throughout.
- On **CUDA GPU**: models load in **float16** (half precision) to reduce VRAM usage
  on memory-constrained GPUs (e.g., 4 GB). The ONNX export captures the model at
  its working precision.
- After export, the Android app can convert to float16 at runtime via the DML or
  CPU execution provider.
- Dynamic axes are set for batch, spatial, and temporal dimensions where
  applicable.
- The VAE encoder handles arbitrary H/W (must be multiples of 32).
- The UNet operates on 9-frame sliding windows for temporal consistency.
- The Turbo-VAED decoder processes all 17 frames in a single batch.

## Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `CUDA out of memory` | Large model + small GPU (4 GB) | Use `--device cpu`, or reduce spatial resolution with `--height`/`--width` | | `onnxruntime-gpu` import error | Conflicting onnxruntime installation | Run `pip uninstall onnxruntime` then `pip install onnxruntime-gpu` |
| `snapshot_download` fails | No internet / HF token missing | Set `HF_TOKEN` env var or use `huggingface-cli login` |
| `ONNX export` hangs | Large model trace | Add `--verbose` to see progress |
| Shape mismatch | Different model revision | Check `config.json` for expected shapes |

## References

- [MobileI2V on HuggingFace](https://huggingface.co/hustvl/MobileI2V)
- [LTX-Video by Lightricks](https://huggingface.co/Lightricks/LTX-Video)
- [Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B)
- [ONNX Runtime](https://onnxruntime.ai/)
