#!/usr/bin/env python3
"""
convert_vae_encoder.py — Export VAE Encoder from LTX-Video to ONNX.

Downloads the LTX-Video VAE from HuggingFace (Lightricks/LTX-Video),
extracts the encoder portion, and exports it as ``vae_encoder.onnx``.

Input:   [1, 3, H, W] float32  — RGB image (H, W any multiple of 32)
Output:  [1, 128, H/8, W/8] float32 — spatial latent (compressed 8x, 128-ch LTX-Video VAE)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

# Ensure the parent of common/ is on sys.path so the common package is importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common.model_utils import MODEL_CACHE_DIR, get_device, get_torch_dtype, export_onnx, verify_onnx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAE_REPO_ID = "Lightricks/LTX-Video"
MODEL_NAME = "vae_encoder"
INPUT_SHAPE = (1, 3, 720, 1280)  # default: 720p
DYNAMIC_HW = {2: "height", 3: "width"}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class VAEEncoderWrapper(torch.nn.Module):
    """Wraps the VAE encoder for standalone ONNX export.

    ``AutoencoderKLLTXVideo`` is a 3D (video) VAE that expects 5D input
    [B, C, F, H, W].  This wrapper handles the 4D→5D→4D conversion so
    the ONNX model accepts standard 4D image tensors [B, C, H, W]:
      1. Unsqueeze frame dimension  [B, C, H, W] → [B, C, 1, H, W]
      2. Run ``vae.encode()`` → DiagonalGaussianDistribution
      3. Sample from the distribution
      4. Multiply by the scaling factor
      5. Squeeze frame dimension   [B, C, 1, H', W'] → [B, C, H', W']
    """

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # AutoencoderKLLTXVideo is a 3D (video) VAE — expects 5D [B, C, F, H, W]
        # Add a singleton frame dimension for single-image encoding
        x = pixel_values.unsqueeze(2)          # [B, C, 1, H, W]

        # Encode returns AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution)
        dist = self.vae.encode(x)

        # `.latent_dist` is the DiagonalGaussianDistribution
        latents = dist.latent_dist.sample()     # [B, 128, 1, H/8, W/8]

        # Apply the scaling factor used during training
        latents = latents * self.vae.config.scaling_factor

        # Remove the singleton frame dimension for the ONNX output
        return latents.squeeze(2)               # [B, 128, H/8, W/8]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_vae_encoder(
    output_dir: Path,
    *,
    device: torch.device,
    height: int = 720,
    width: int = 1280,
    verbose: bool = False,
) -> Path:
    """Download the LTX-Video VAE and export just the encoder to ONNX.

    Args:
        output_dir: Directory to write ``vae_encoder.onnx`` into.
        device:     Target device (cuda or cpu).
        height:     Example image height (default 720).
        width:      Example image width  (default 1280).
        verbose:    Print ONNX export progress.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    logger.info("=== VAE Encoder Conversion ===")
    logger.info("Loading VAE from %s ...", VAE_REPO_ID)
    torch_dtype = get_torch_dtype(device)

    # Load the VAE from the diffusers pipeline (avoids loading the full model)
    from diffusers import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_pretrained(
        VAE_REPO_ID,
        subfolder="vae",
        torch_dtype=torch_dtype,
        cache_dir=MODEL_CACHE_DIR,
    )
    vae.to(device)
    vae.eval()
    logger.info("VAE loaded (%.2fM params)", sum(p.numel() for p in vae.parameters()) / 1e6)

    # Wrap encoder
    wrapper = VAEEncoderWrapper(vae).to(device)
    wrapper.eval()

    # Build dummy inputs
    dummy = torch.randn(1, 3, height, width, dtype=torch_dtype, device=device)

    dummy_inputs = {"pixel_values": dummy}
    input_names = ["pixel_values"]
    output_names = ["latent"]

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "pixel_values": {0: "batch", **DYNAMIC_HW},
        "latent":       {0: "batch", 2: "latent_height", 3: "latent_width"},
    }

    # Export
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

    # Quick verification with onnxruntime
    feeds = {
        "pixel_values": dummy.cpu().numpy(),
    }
    verify_onnx(onnx_path, feeds=feeds, expected_output_names=output_names, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VAE Encoder to ONNX")
    parser.add_argument("--output", "-o", type=Path, default=Path("./models"),
                        help="Output directory (default: ./models)")
    parser.add_argument("--height", type=int, default=720,
                        help="Example image height (default: 720)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Example image width (default: 1280)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print export progress")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda / cpu)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    device = torch.device(args.device) if args.device else get_device()

    convert_vae_encoder(
        output_dir=args.output,
        device=device,
        height=args.height,
        width=args.width,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
