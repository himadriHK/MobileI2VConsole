"""
ONNX Runtime configuration and optimization helpers for MobileI2V models.

Provides:
  - optimize_onnx_model:  Apply graph optimization level.
  - convert_to_fp16:      Convert float32 ONNX to float16 (where possible).
  - create_session_options: Build onnxruntime SessionOptions for Android.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import onnx
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, SessionOptions, ExecutionMode

logger = logging.getLogger(__name__)


def optimize_onnx_model(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    optimization_level: GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
) -> Path:
    """Apply ONNX graph optimization and save to a new file.

    Args:
        onnx_path:          Path to the input ``.onnx`` file.
        output_path:        Output path (defaults to ``<stem>_optimized.onnx``).
        optimization_level: ONNX Runtime graph optimization level.

    Returns:
        Path to the optimized model.
    """
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"

    logger.info("Optimizing %s (level=%s) -> %s", onnx_path, optimization_level, output_path)

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    # ONNX graph optimization via onnxruntime's inline optimizer
    from onnxruntime.transformers import optimizer as t_optimizer

    opt = t_optimizer.optimize_model(
        str(onnx_path),
        model_type="bert",  # generic; works for non-BERT models too
        num_heads=0,
        hidden_size=0,
        optimization_level=optimization_level,
    )
    opt.save_model_to_file(str(output_path))
    logger.info("Optimized model saved to %s", output_path)
    return output_path


def convert_to_fp16(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
) -> Path:
    """Convert float32 ONNX model weights to float16.

    Uses the ONNX converter to cast initializers (weights/biases) from FP32 to FP16.
    Activations remain FP32 (runtime cast is handled by the execution provider).

    Args:
        onnx_path:        Path to the input ``.onnx`` file.
        output_path:      Output path (defaults to ``<stem>_fp16.onnx``).
        min_positive_val: Minimum positive value to avoid underflow.
        max_finite_val:   Maximum finite value to avoid overflow.

    Returns:
        Path to the FP16 model.
    """
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_fp16.onnx"

    logger.info("Converting %s to float16 -> %s", onnx_path, output_path)

    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(onnx_path))
    model_fp16 = convert_float_to_float16(
        model,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        keep_io_types=True,  # keep inputs/outputs as FP32
    )
    onnx.save(model_fp16, str(output_path))
    logger.info("FP16 conversion saved to %s", output_path)
    return output_path


def create_session_options(
    intra_op_threads: int = 2,
    inter_op_threads: int = 2,
    enable_mem_pattern: bool = True,
    enable_cpu_mem_arena: bool = True,
) -> SessionOptions:
    """Build an onnxruntime SessionOptions configured for mobile/Android.

    Args:
        intra_op_threads:     Number of threads for intra-op parallelism.
        inter_op_threads:     Number of threads for inter-op parallelism.
        enable_mem_pattern:   Enable memory pattern optimization.
        enable_cpu_mem_arena: Enable CPU memory arena.

    Returns:
        Configured onnxruntime ``SessionOptions``.
    """
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra_op_threads
    opts.inter_op_num_threads = inter_op_threads
    opts.enable_mem_pattern = enable_mem_pattern
    opts.enable_cpu_mem_arena = enable_cpu_mem_arena
    opts.execution_mode = ExecutionMode.ORT_SEQUENTIAL
    return opts
