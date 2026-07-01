#!/usr/bin/env python3
"""
convert_all.py — Orchestrate all four MobileI2V model conversions.

Sequentially converts each model to ONNX, validates with onnxruntime,
computes SHA256 hashes, and produces a ``models_manifest.json``.

Usage:
    python convert_all.py --output ./models/
    python convert_all.py --output ./models/ --optimize   # also exports .ort
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure the parent of common/ is on sys.path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from common.model_utils import compute_ort_export, get_device, enable_cuda_optimizations

# Import converters
from convert_vae_encoder import convert_vae_encoder
from convert_qwen2_encoder import convert_qwen2_encoder
from convert_mobilei2v_unet import convert_mobilei2v_unet
from convert_turbo_vaed import convert_turbo_vaed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256sum(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(onnx_files: List[Path], ort_files: List[Path]) -> Dict[str, Any]:
    """Build a JSON-serializable manifest of all exported models."""
    manifest: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {},
    }

    for path in onnx_files:
        manifest["models"][path.stem] = {
            "format": "onnx",
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256sum(path),
        }

    for path in ort_files:
        key = path.stem.replace("_optimized", "")
        if key not in manifest["models"]:
            manifest["models"][key] = {}
        manifest["models"][key]["ort_path"] = str(path)
        manifest["models"][key]["ort_size_bytes"] = path.stat().st_size
        manifest["models"][key]["ort_sha256"] = sha256sum(path)

    return manifest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert all MobileI2V models to ONNX (and optionally ORT format)."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./models"),
        help="Output directory for ONNX models (default: ./models)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Also export to ORT mobile-optimized format",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda / cpu)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed export progress",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else get_device()
    enable_cuda_optimizations(device)

    logger.info("=" * 60)
    logger.info("MobileI2V — ONNX Conversion Pipeline")
    logger.info("Output directory: %s", output_dir.resolve())
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: VAE Encoder
    # ------------------------------------------------------------------
    logger.info("\n>>> Phase 1/4: VAE Encoder")
    t0 = time.time()
    vae_encoder_path = convert_vae_encoder(
        output_dir=output_dir,
        device=device,
        verbose=args.verbose,
    )
    logger.info("<<< VAE Encoder done in %.1f s\n", time.time() - t0)

    # ------------------------------------------------------------------
    # Phase 2: Qwen2 Text Encoder
    # ------------------------------------------------------------------
    logger.info(">>> Phase 2/4: Qwen2-0.5B Text Encoder")
    t0 = time.time()
    qwen2_path = convert_qwen2_encoder(
        output_dir=output_dir,
        device=device,
        verbose=args.verbose,
    )
    logger.info("<<< Qwen2 Encoder done in %.1f s\n", time.time() - t0)

    # ------------------------------------------------------------------
    # Phase 3: MobileI2V UNet
    # ------------------------------------------------------------------
    logger.info(">>> Phase 3/4: MobileI2V UNet")
    t0 = time.time()
    unet_path = convert_mobilei2v_unet(
        output_dir=output_dir,
        device=device,
        verbose=args.verbose,
    )
    logger.info("<<< MobileI2V UNet done in %.1f s\n", time.time() - t0)

    # ------------------------------------------------------------------
    # Phase 4: Turbo-VAED Decoder
    # ------------------------------------------------------------------
    logger.info(">>> Phase 4/4: Turbo-VAED Decoder")
    t0 = time.time()
    vaed_path = convert_turbo_vaed(
        output_dir=output_dir,
        device=device,
        verbose=args.verbose,
    )
    logger.info("<<< Turbo-VAED Decoder done in %.1f s\n", time.time() - t0)

    # ------------------------------------------------------------------
    # Post-processing: ORT export
    # ------------------------------------------------------------------
    onnx_files = [vae_encoder_path, qwen2_path, unet_path, vaed_path]
    ort_files: List[Path] = []

    if args.optimize:
        logger.info("Exporting to ORT mobile-optimized format ...")
        for onnx_path in onnx_files:
            logger.info("  Converting %s ...", onnx_path.name)
            try:
                ort_path = compute_ort_export(onnx_path, output_dir)
                ort_files.append(ort_path)
            except Exception as exc:
                logger.error("  Failed: %s", exc)
        logger.info("ORT export complete.\n")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = build_manifest(onnx_files, ort_files)
    manifest_path = output_dir / "models_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Conversion complete.  Summary:")
    logger.info("=" * 60)
    for stem, info in manifest["models"].items():
        size_mb = info["size_bytes"] / (1024 * 1024)
        logger.info("  %-25s %8.2f MB  sha256:%s", stem, size_mb, info["sha256"][:16])
        if "ort_path" in info:
            ort_size_mb = info["ort_size_bytes"] / (1024 * 1024)
            logger.info("  %-25s %8.2f MB  (ORT)  sha256:%s",
                        stem + " [ort]", ort_size_mb, info["ort_sha256"][:16])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
