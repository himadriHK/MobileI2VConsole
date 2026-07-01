#!/usr/bin/env python3
"""
convert_mobilei2v_unet.py — MobileI2V DiT (Mobiledit) → ONNX converter.

Downloads the hybrid_371.pth checkpoint from HuggingFace, creates the
Mobiledit model (vendored at scripts/convert/models/mobiledit.py),
loads only the DiT weights, wraps for ONNX, and exports with
export_onnx() from common/model_utils.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import _pickle

import torch
from huggingface_hub import hf_hub_download

# Add models/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from mobiledit import MobileditONNXWrapper, mobiledit_300m_P1_D16
from common.model_utils import (
    MODEL_CACHE_DIR,
    enable_cuda_optimizations,
    export_onnx,
    get_device,
    get_torch_dtype,
    verify_onnx,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------

def download_checkpoint(cache_dir: str) -> str:
    """Download hybrid_371.pth from HuggingFace, return local path."""
    logger.info("Downloading hybrid_371.pth from hustvl/MobileI2V ...")
    path = hf_hub_download(
        repo_id="hustvl/MobileI2V",
        filename="hybrid_371.pth",
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )
    logger.info("Downloaded checkpoint to: %s", path)
    return path


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(device: torch.device, torch_dtype: torch.dtype) -> torch.nn.Module:
    """Create the Mobiledit-300M model with flash attention (ONNX-compatible)."""
    logger.info("Creating Mobiledit-300M model (attn_type='flash') ...")
    model = mobiledit_300m_P1_D16(attn_type='flash')
    model = model.to(device=device, dtype=torch_dtype)
    logger.info(
        "Model created (%.2fM params)",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load DiT weights from the hybrid_371.pth checkpoint.

    The checkpoint has structure
    ``{"state_dict": {...}, "epoch": ..., "rng_state": ...}``
    and may contain extra keys (guide_image embedder, optimizer states)
    that are not part of the Mobiledit model.  Only keys present in
    ``model.state_dict()`` are loaded.

    If ``pos_embed`` shape mismatches (different spatial resolution
    during training), it is skipped so the model keeps its own
    reinitialized value.
    """
    logger.info("Loading checkpoint from %s ...", checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except _pickle.UnpicklingError:
        logger.warning(
            "weights_only=True failed — checkpoint contains non-tensor metadata. "
            "Falling back to weights_only=False. "
            "Only proceed if the checkpoint source (hustvl/MobileI2V) is trusted."
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_state = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()

    # Filter to only keys that exist in the model
    filtered: dict[str, torch.Tensor] = {}
    skipped_pos_embed = False
    for key, value in ckpt_state.items():
        if key not in model_state:
            logger.debug("  Skipping extra key: %s", key)
            continue
        # Handle pos_embed size mismatch (different spatial resolution)
        if key == "pos_embed" and value.shape != model_state[key].shape:
            logger.warning(
                "pos_embed shape mismatch: checkpoint %s vs model %s — skipping",
                value.shape, model_state[key].shape,
            )
            skipped_pos_embed = True
            continue
        filtered[key] = value.to(device=device)

    extra_count = len(ckpt_state) - len(filtered) - (1 if skipped_pos_embed else 0)
    logger.info(
        "Loaded %d / %d keys (skipped %d extra, %s pos_embed mismatch)",
        len(filtered),
        len(model_state),
        extra_count,
        "1" if skipped_pos_embed else "0",
    )

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        logger.warning("Missing keys (not in checkpoint): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys (not in model): %s", unexpected)

    logger.info("Checkpoint loaded successfully.")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    verbose: bool,
) -> str:
    """Wrap the model and export to ONNX."""
    wrapper = MobileditONNXWrapper(model).to(device=device, dtype=torch_dtype)
    wrapper.eval()

    # Dummy inputs: batch=2, 17 frames, 300 text tokens (896-dim Qwen2), 32x32 spatial, 128 latent channels
    latent = torch.randn(2, 128, 17, 32, 32, device=device, dtype=torch_dtype)
    text_emb = torch.randn(2, 300, 896, device=device, dtype=torch_dtype)
    timestep = torch.randint(0, 1000, (2,), device=device, dtype=torch.long)

    dummy_inputs = {
        'latent': latent,
        'text_emb': text_emb,
        'timestep': timestep,
    }
    input_names = ['latent', 'text_emb', 'timestep']
    output_names = ['denoised_latent']

    dynamic_axes = {
        'latent': {0: 'batch', 3: 'height', 4: 'width'},
        'text_emb': {0: 'batch', 1: 'text_tokens'},
        'timestep': {0: 'batch'},
        'denoised_latent': {0: 'batch', 3: 'height', 4: 'width'},
    }

    output_dir = str(Path(output_path).parent)
    model_name = Path(output_path).stem

    onnx_path = export_onnx(
        model=wrapper,
        model_name=model_name,
        output_dir=output_dir,
        dummy_inputs=dummy_inputs,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        verbose=verbose,
    )
    logger.info("ONNX exported to: %s", onnx_path)

    # Verification with onnxruntime
    feeds = {
        'latent': latent.cpu().numpy(),
        'text_emb': text_emb.cpu().numpy(),
        'timestep': timestep.cpu().numpy(),
    }
    verify_onnx(
        onnx_path,
        feeds=feeds,
        expected_output_names=output_names,
    )

    return str(onnx_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MobileI2V DiT (Mobiledit) to ONNX",
    )
    parser.add_argument(
        '--output', '-o', type=str, default='mobilei2v_unet.onnx',
        help='Output ONNX file path (default: mobilei2v_unet.onnx)',
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Override device (cuda / cpu)',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print export progress',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(message)s',
    )

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    torch_dtype = get_torch_dtype(device)

    if device.type == 'cuda':
        enable_cuda_optimizations(device)

    logger.info("Device: %s  |  dtype: %s", device, torch_dtype)

    # Pipeline
    checkpoint_path = download_checkpoint(str(MODEL_CACHE_DIR))
    model = create_model(device, torch_dtype)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    export_to_onnx(model, args.output, device, torch_dtype, args.verbose)

    logger.info("=== Conversion complete ===")


# ---------------------------------------------------------------------------
# Pipeline entry point (for convert_all.py)
# ---------------------------------------------------------------------------

def convert_mobilei2v_unet(
    output_dir: Path,
    *,
    device: torch.device,
    verbose: bool = False,
) -> Path:
    """Full pipeline: download checkpoint → create model → load weights → export ONNX.

    Args:
        output_dir: Directory to write ``mobilei2v_unet.onnx`` into.
        device:     Target device (cuda or cpu).
        verbose:    Print ONNX export progress.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    logger.info("=== MobileI2V UNet (Mobiledit) Conversion ===")
    torch_dtype = get_torch_dtype(device)

    if device.type == 'cuda':
        enable_cuda_optimizations(device)

    logger.info("Device: %s  |  dtype: %s", device, torch_dtype)

    checkpoint_path = download_checkpoint(str(MODEL_CACHE_DIR))
    model = create_model(device, torch_dtype)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    output_path = str(output_dir / 'mobilei2v_unet.onnx')
    onnx_path = export_to_onnx(model, output_path, device, torch_dtype, verbose)
    return Path(onnx_path)


if __name__ == '__main__':
    main()
