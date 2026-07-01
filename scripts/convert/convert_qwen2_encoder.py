#!/usr/bin/env python3
"""
convert_qwen2_encoder.py — Export Qwen2-0.5B Text Encoder to ONNX.

Downloads Qwen2-0.5B from HuggingFace, extracts the base transformer
(without LM head), and exports it as ``qwen2_encoder.onnx``.

Inputs:  ``input_ids`` [1, seq_len] int64
         ``attention_mask`` [1, seq_len] int64
Output:  ``last_hidden_state`` [1, seq_len, 1024] float32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

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

MODEL_ID = "Qwen/Qwen2-0.5B"
MODEL_NAME = "qwen2_encoder"
MAX_SEQ_LEN = 77  # default prompt token length used in I2V pipelines


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class Qwen2EncoderWrapper(torch.nn.Module):
    """Wraps the Qwen2 base model to output last hidden state.

    Strips the LM head so export contains only the encoder stack.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.last_hidden_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_qwen2_encoder(
    output_dir: Path,
    *,
    device: torch.device,
    max_seq_len: int = MAX_SEQ_LEN,
    verbose: bool = False,
) -> Path:
    """Download Qwen2-0.5B and export the text encoder to ONNX.

    Args:
        output_dir:   Directory to write ``qwen2_encoder.onnx`` into.
        device:       Target device (cuda or cpu).
        max_seq_len:  Maximum sequence length for dynamic axis bounds.
        verbose:      Print ONNX export progress.

    Returns:
        Path to the exported ``.onnx`` file.
    """
    logger.info("=== Qwen2-0.5B Text Encoder Conversion ===")
    logger.info("Loading %s ...", MODEL_ID)
    torch_dtype = get_torch_dtype(device)

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",       # use PyTorch's scaled_dot_product_attention
        device_map=None,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded (%.2fM params)", sum(p.numel() for p in model.parameters()) / 1e6)

    # Wrap and move
    wrapper = Qwen2EncoderWrapper(model).to(device)
    wrapper.eval()

    # Build dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (1, max_seq_len), dtype=torch.long, device=device)
    dummy_attn_mask = torch.ones(1, max_seq_len, dtype=torch.long, device=device)

    dummy_inputs = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attn_mask,
    }
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]

    dynamic_axes: Dict[str, Dict[int, str]] = {
        "input_ids":        {0: "batch", 1: "sequence"},
        "attention_mask":   {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
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

    # Verification
    feeds = {
        "input_ids": dummy_input_ids.cpu().numpy(),
        "attention_mask": dummy_attn_mask.cpu().numpy(),
    }
    verify_onnx(onnx_path, feeds=feeds, expected_output_names=output_names, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen2-0.5B Text Encoder to ONNX")
    parser.add_argument("--output", "-o", type=Path, default=Path("./models"),
                        help="Output directory (default: ./models)")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help="Maximum sequence length for export (default: 77)")
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

    convert_qwen2_encoder(
        output_dir=args.output,
        device=device,
        max_seq_len=args.max_seq_len,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
