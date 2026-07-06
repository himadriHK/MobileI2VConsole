#!/usr/bin/env python3
"""
convert_turbo_vaed.py — Export Turbo-VAED Decoder to ONNX.

Downloads the config from GitHub (hustvl/Turbo-VAED/configs/) and the
checkpoint from HuggingFace (hustvl/Turbo-VAED), then builds the decoder
using the vendored turbo_vaed_model.py and exports to ONNX.

The Turbo-VAED is a decoder-only video VAE that converts latent video frames
back to RGB frames.  It is *not* a diffusers AutoencoderKL — the vendored
model code is required.

Input:   [B, latent_channels, T, H, W]  float32 — latent video tensor
Output:  [B, 3, T*8, H*32, W*32]        float32 — decoded RGB frames

Variants (latent_channels — default LTX):
  - LTX:          latent_channels=128, patch_size=4  (smallest, MobileI2V-relevant)
  - CogVideo5B:   latent_channels=16,  patch_size=1
  - HunyuanVideo: latent_channels=16,  patch_size=1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import torch

# Ensure the parent of common/ is on sys.path so packages are importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common.model_utils import MODEL_CACHE_DIR, get_device, get_torch_dtype, export_onnx, verify_onnx
from models.turbo_vaed_model import build_turbo_vaed_decoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAED_HF_REPO = "hustvl/Turbo-VAED"
VAED_CONFIG_BASE = "https://raw.githubusercontent.com/hustvl/Turbo-VAED/main/configs"
MODEL_NAME = "turbo_vaed"

VARIANTS = {
    "LTX": {
        "config_file": "Turbo-VAED-LTX.json",
        "checkpoint_file": "Turbo-VAED-LTX.pth",
        "latent_channels": 128,
        "dummy_t": 5,
        "dummy_h": 23,
        "dummy_w": 40,
    },
    "CogVideo5B": {
        "config_file": "Turbo-VAED-CogVideo5B.json",
        "checkpoint_file": "Turbo-VAED-CogVideo5B.pth",
        "latent_channels": 16,
        "dummy_t": 5,
        "dummy_h": 23,
        "dummy_w": 40,
    },
    "HunyuanVideo": {
        "config_file": "Turbo-VAED-HunyuanVideo.json",
        "checkpoint_file": "Turbo-VAED-HunyuanVideo.pth",
        "latent_channels": 16,
        "dummy_t": 5,
        "dummy_h": 23,
        "dummy_w": 40,
    },
}
DEFAULT_VARIANT = "LTX"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class TurboVAEDWrapper(torch.nn.Module):
    """Wraps the Turbo-VAED decoder for standalone ONNX export.

    The decoder itself is a TurboVAEDDecoder3d — this wrapper simply exposes
    ``.forward(latent) -> frames`` for clean ONNX graph naming.
    """

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_turbo_vaed(
    output_dir: Path,
    *,
    device: torch.device,
    variant: str = DEFAULT_VARIANT,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: Optional[int] = None,
    verbose: bool = False,
) -> Path:
    """Download Turbo-VAED config + checkpoint and export decoder to ONNX.

    Args:
        output_dir: Directory to write ``turbo_vaed.onnx`` into.
        device:     Target device (cuda or cpu).
        variant:    Model variant (LTX, CogVideo5B, HunyuanVideo).
        height:     Latent spatial height (default from variant, e.g. 23 for LTX).
        width:      Latent spatial width  (default from variant, e.g. 40 for LTX).
        num_frames: Number of latent frames for dummy input (default from variant, e.g. 5).
        verbose:    Print ONNX export progress.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    logger.info("=== Turbo-VAED Decoder Conversion (%s) ===", variant)

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(VARIANTS.keys())}")

    vinfo = VARIANTS[variant]
    config_file = vinfo["config_file"]
    checkpoint_file = vinfo["checkpoint_file"]
    latent_channels = vinfo["latent_channels"]

    dummy_t = num_frames or vinfo["dummy_t"]
    dummy_h = height or vinfo["dummy_h"]
    dummy_w = width or vinfo["dummy_w"]

    torch_dtype = get_torch_dtype(device)

    # ------------------------------------------------------------------
    # Step 1: Download config from GitHub
    # ------------------------------------------------------------------
    config_url = f"{VAED_CONFIG_BASE}/{config_file}"
    logger.info("Downloading config from %s ...", config_url)
    with urllib.request.urlopen(config_url) as resp:
        config = json.loads(resp.read().decode())
    logger.info("Config loaded: %d top-level keys", len(config))

    # Verify latent_channels consistency
    config_latent = config.get("latent_channels", None)
    if config_latent is not None and config_latent != latent_channels:
        logger.warning(
            "Config latent_channels=%d differs from variant default %d. "
            "Using config value.",
            config_latent, latent_channels,
        )
        latent_channels = config_latent

    # ------------------------------------------------------------------
    # Step 2: Download checkpoint from HuggingFace
    # ------------------------------------------------------------------
    from huggingface_hub import hf_hub_download

    local_cache = MODEL_CACHE_DIR / "turbo_vaed"
    local_cache.mkdir(parents=True, exist_ok=True)

    ckpt_path = hf_hub_download(
        repo_id=VAED_HF_REPO,
        filename=checkpoint_file,
        local_dir=local_cache,
        resume_download=True,
    )
    logger.info("Checkpoint: %s", ckpt_path)

    # ------------------------------------------------------------------
    # Step 3: Build decoder from config
    # ------------------------------------------------------------------
    logger.info("Building Turbo-VAED decoder from config ...")
    decoder = build_turbo_vaed_decoder(config)
    logger.info(
        "Decoder built: %.2fM params (unloaded)",
        sum(p.numel() for p in decoder.parameters()) / 1e6,
    )

    # ------------------------------------------------------------------
    # Step 4: Load state dict (strip "decoder." prefix)
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint ...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # The full model state dict has keys like "decoder.conv_in.conv.weight".
    # Our standalone decoder expects "conv_in.conv.weight".  Strip the prefix.
    decoder_keys: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("decoder."):
            decoder_keys[k[len("decoder."):]] = v

    if not decoder_keys:
        logger.warning(
            "No keys with 'decoder.' prefix found — trying raw state dict"
        )
        decoder_keys = state_dict

    missing, unexpected = decoder.load_state_dict(decoder_keys, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing[:8])
    if unexpected:
        logger.info("Unexpected keys (expected — encoder/non-decoder): %d", len(unexpected))

    decoder.to(device, dtype=torch_dtype)
    decoder.eval()
    logger.info(
        "Decoder loaded on %s (%.2fM params)",
        device, sum(p.numel() for p in decoder.parameters()) / 1e6,
    )

    # ------------------------------------------------------------------
    # Step 5: Wrap and export
    # ------------------------------------------------------------------
    wrapper = TurboVAEDWrapper(decoder).to(device, dtype=torch_dtype)
    wrapper.eval()

    # Build dummy input: [B, C, T, H, W]
    dummy_latent = torch.randn(
        1, latent_channels, dummy_t, dummy_h, dummy_w,
        dtype=torch_dtype, device=device,
    )

    dummy_inputs = {"latent": dummy_latent}
    input_names = ["latent"]
    output_names = ["frames"]

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "latent": {0: "batch", 2: "num_frames", 3: "height", 4: "width"},
        "frames": {0: "batch", 2: "num_frames", 3: "height", 4: "width"},
    }

    logger.info(
        "Dummy input shape: %s (B=%d, C=%d, T=%d, H=%d, W=%d)",
        list(dummy_latent.shape), *dummy_latent.shape,
    )

    onnx_path = export_onnx(
        model=wrapper,
        model_name=MODEL_NAME,
        output_dir=output_dir,
        dummy_inputs=dummy_inputs,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Step 6: Verify
    # ------------------------------------------------------------------
    feeds = {"latent": dummy_latent.cpu().numpy()}
    verify_onnx(
        onnx_path,
        feeds=feeds,
        expected_output_names=output_names,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    return onnx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Turbo-VAED Decoder to ONNX",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("./models"),
        help="Output directory (default: ./models)",
    )
    parser.add_argument(
        "--variant", type=str, default=DEFAULT_VARIANT,
        choices=list(VARIANTS.keys()),
        help=f"Model variant (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Latent height for dummy input (default: variant default)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Latent width for dummy input (default: variant default)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="Number of latent frames for dummy input (default: variant default)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print export progress",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda / cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    device = torch.device(args.device) if args.device else get_device()

    convert_turbo_vaed(
        output_dir=args.output,
        device=device,
        variant=args.variant,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
