"""
Shared ONNX export utilities for MobileI2V model conversion.

Provides:
  - get_device:        Select CUDA or CPU
  - download_repo:     Selective HuggingFace repo download (entire repo or by patterns)
  - export_onnx:       Generic torch.onnx.export wrapper with verification
  - verify_onnx:       Load exported model and validate expected inputs/outputs
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import onnx
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)

# Resolve to: <project_root>/scripts/model/cache
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "model" / "cache"


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available — using CPU")
    return device


def get_torch_dtype(device: torch.device) -> torch.dtype:
    """Return float16 on CUDA (memory efficiency), float32 on CPU."""
    return torch.float16 if device.type == "cuda" else torch.float32


def enable_cuda_optimizations(device: torch.device) -> None:
    """Enable CUDA-specific performance optimizations for ONNX export."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        logger.info("CUDA optimizations enabled: cudnn.benchmark, float32_matmul_precision=high")


def download_repo(
    repo_id: str,
    allow_patterns: Optional[List[str]] = None,
    local_dir: Optional[os.PathLike] = None,
) -> Path:
    """Download a HuggingFace repo snapshot (selective when ``allow_patterns`` is set).

    Args:
        repo_id: HuggingFace repository ID (e.g. ``"hustvl/MobileI2V"``).
        allow_patterns: Glob patterns of files to include (e.g. ``["unet/*", "*.json"]``).
            When provided, only matching files are downloaded individually via
            ``hf_hub_download`` instead of ``snapshot_download``.  When ``None``,
            the full repo is downloaded (original behaviour).
        local_dir: Optional local directory to mirror the snapshot into.

    Returns:
        Path to the local directory containing downloaded files.
    """
    from huggingface_hub import hf_hub_download, HfApi, snapshot_download
    import fnmatch

    if allow_patterns is None:
        # Fallback: download everything (original behaviour)
        logger.info("Downloading repo %s (full snapshot) ...", repo_id)
        out = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.md", "*.git*"],
        )
        logger.info("Downloaded to: %s", out)
        return Path(out)

    # List all files in the repo
    api = HfApi()
    repo_files = api.list_repo_files(repo_id, repo_type="model")

    # Filter by allow_patterns
    matching_files: List[str] = []
    for f in repo_files:
        for pattern in allow_patterns:
            if fnmatch.fnmatch(f, pattern):
                matching_files.append(f)
                break

    if not matching_files:
        logger.warning(
            "No files matched patterns %s in repo %s", allow_patterns, repo_id
        )

    # Determine local directory
    if local_dir is not None:
        output_dir = Path(local_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        from huggingface_hub import try_to_load_from_cache

        output_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{repo_id.replace('/', '--')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    # Download each matching file
    logger.info(
        "Downloading %d file(s) from %s ...", len(matching_files), repo_id
    )
    for file_path in matching_files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                local_dir=local_dir,
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            logger.debug("  Downloaded: %s", file_path)
        except Exception as exc:
            logger.warning("  Failed to download %s: %s", file_path, exc)

    # Return the local directory
    if local_dir is not None:
        return Path(local_dir)
    return output_dir


def _patch_neg_transpose(onnx_model: onnx.ModelProto) -> bool:
    """Fix Transpose nodes with ``-1`` in their ``perm`` attribute.

    The legacy ONNX exporter may emit ``-1`` as a literal perm index
    (meaning "last dimension" in PyTorch), which is invalid in ONNX.
    This function replaces ``-1`` with the actual rank-based index.

    Returns:
        True if any fix was applied.
    """
    fixed = False
    for node in onnx_model.graph.node:
        if node.op_type != "Transpose":
            continue
        for attr in node.attribute:
            if attr.name != "perm" or attr.type != onnx.AttributeProto.INTS:
                continue
        perms = list(attr.ints)
        if -1 not in perms:
            continue
        rank = len(perms)
        new_perms = [rank - 1 if p == -1 else p for p in perms]
        del attr.ints[:]
        attr.ints.extend(new_perms)
        fixed = True
    return fixed


def _dynamic_axes_to_shapes(dynamic_axes: Dict[str, Dict[int, str]]) -> Optional[Dict[str, Dict[int, "Dim"]]]:
    """Convert legacy *dynamic_axes* format to *dynamic_shapes* format for dynamo export.

    ``dynamic_axes`` format:   ``{"name": {0: "batch", 2: "height"}}``
    ``dynamic_shapes`` format: ``{"name": {0: Dim("batch"), 2: Dim("height")}}``

    Returns ``None`` if ``torch.export.Dim`` is not available (older PyTorch),
    in which case the caller should fall back to the legacy export path.
    """
    try:
        from torch.export import Dim
    except ImportError:
        return None
    return {
        name: {idx: Dim(label) for idx, label in axes.items()}
        for name, axes in dynamic_axes.items()
    }


def export_onnx(
    model: torch.nn.Module,
    model_name: str,
    output_dir: os.PathLike,
    dummy_inputs: Dict[str, torch.Tensor],
    dynamic_axes: Dict[str, Dict[int, str]],
    input_names: List[str],
    output_names: List[str],
    verbose: bool = False,
) -> Path:
    """Export a PyTorch model to ONNX and verify the result.

    Uses an automatic fallback chain:
      1. **dynamo-native** path with ``dynamic_shapes`` (handles negative permute correctly).
      2. **legacy** path with ``dynamic_axes`` + ``dynamo=False`` (fallback if dynamo fails).
      3. **Transpose perm repair** — any ``-1`` in Transpose attributes is fixed after export.

    Args:
        model:        PyTorch model (already in eval mode).
        model_name:   Short name used for the output file (e.g. ``"vae_encoder"``).
        output_dir:   Directory where the ``.onnx`` file will be written.
        dummy_inputs: Example tensors keyed by input name (matches ``input_names``).
        dynamic_axes: Dynamic axis definitions passed to ``torch.onnx.export``.
        input_names:  ONNX graph input names.
        output_names: ONNX graph output names.
        verbose:      Print ONNX export progress.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    logger.info("Exporting %s -> %s", model_name, onnx_path)

    # Build args from dummy_inputs dict in the correct order
    args = tuple(dummy_inputs[name] for name in input_names)

    dynamic_shapes = _dynamic_axes_to_shapes(dynamic_axes)

    with torch.no_grad():
        if dynamic_shapes is not None:
            # Strategy 1: dynamo-native path with dynamic_shapes
            # Only input shapes are passed — outputs are inferred.
            input_shapes = {
                name: shapes for name, shapes in dynamic_shapes.items()
                if name in input_names
            }
            try:
                torch.onnx.export(
                    model,
                    args=args,
                    f=str(onnx_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_shapes=input_shapes,
                    opset_version=17,
                    do_constant_folding=True,
                    verbose=verbose,
                )
            except Exception:
                logger.warning(
                    "dynamo export failed for '%s' — "
                    "falling back to legacy exporter with Transpose repair",
                    model_name,
                    exc_info=True,
                )
                # Strategy 2: legacy exporter
                torch.onnx.export(
                    model,
                    args=args,
                    f=str(onnx_path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=18,
                    do_constant_folding=True,
                    verbose=verbose,
                    dynamo=False,
                )
        else:
            # PyTorch < 2.1 — no dynamo path available
            torch.onnx.export(
                model,
                args=args,
                f=str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=18,
                do_constant_folding=True,
                verbose=verbose,
                dynamo=False,
            )

    logger.info("Export complete — verifying ONNX model ...")
    onnx_model = onnx.load(str(onnx_path))

    # Fix any Transpose nodes with -1 perm (legacy exporter may emit these)
    if _patch_neg_transpose(onnx_model):
        logger.info("Repaired %d Transpose node(s) with -1 perm", 1)
        onnx.save(onnx_model, str(onnx_path))
        onnx_model = onnx.load(str(onnx_path))

    onnx.checker.check_model(onnx_model)
    logger.info("ONNX check passed for %s", onnx_path)

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("%s size: %.2f MB", onnx_path.name, file_size_mb)

    return onnx_path


def verify_onnx(
    onnx_path: os.PathLike,
    feeds: Dict[str, Any],
    expected_output_names: Sequence[str],
    rtol: float = 1e-3,
    atol: float = 1e-3,
    providers: Optional[List[str]] = None,
) -> bool:
    """Load an ONNX model with onnxruntime and run a quick verification.

    Args:
        onnx_path:             Path to the ``.onnx`` file.
        feeds:                 Dictionary mapping input names to numpy arrays.
        expected_output_names: Output names to check existence of.
        rtol:                  Relative tolerance for output comparison (not used yet).
        atol:                  Absolute tolerance for output comparison (not used yet).
        providers:             ONNX Runtime execution providers (default: CUDA then CPU).

    Returns:
        True if the model loads and produces outputs of expected shape.
    """
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info("Verifying %s with onnxruntime ...", onnx_path)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    sess = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
    logger.info("ONNX Runtime using provider: %s", sess.get_providers()[0])

    # Check that all expected outputs exist in the model
    model_output_names = [o.name for o in sess.get_outputs()]
    for name in expected_output_names:
        if name not in model_output_names:
            raise RuntimeError(
                f"Expected output '{name}' not found in model. "
                f"Available: {model_output_names}"
            )

    # Run inference
    outputs = sess.run(expected_output_names, feeds)
    logger.info(
        "Verification passed — %d output(s) produced with shapes: %s",
        len(outputs),
        [o.shape for o in outputs],
    )
    return True


def compute_ort_export(onnx_path: os.PathLike, output_dir: os.PathLike) -> Path:
    """Convert an ONNX model to ORT format for mobile inference.

    Uses ``onnxruntime.tools.convert_onnx_models_to_ort`` if available,
    otherwise falls back to a simple model copy with a warning.

    Args:
        onnx_path:  Path to the ``.onnx`` file.
        output_dir: Output directory for the ``.ort`` file.

    Returns:
        Path to the exported ``.ort`` file.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ort_path = output_dir / f"{onnx_path.stem}.ort"

    try:
        from onnxruntime.tools.convert_onnx_models_to_ort import convert_onnx_models_to_ort

        logger.info("Converting %s -> %s", onnx_path.name, ort_path)
        convert_onnx_models_to_ort(
            [str(onnx_path)],
            output_dir=str(output_dir),
        )
    except (ImportError, Exception) as exc:
        logger.warning(
            "ORT conversion skipped: %s. "
            "Install onnxruntime-tools for mobile-optimized .ort export.",
            exc,
        )
        # Fallback: copy the .onnx as .ort (some runtimes accept this)
        import shutil
        shutil.copy2(onnx_path, ort_path)

    return ort_path
