---
name: pytorch-onnx-debug
description: Systematic debugging of PyTorch ONNX export failures — diagnose and fix common issues with torch.onnx.export, SymInt unhashability, dict caching, dynamic control flow, and dead-code elimination
---

# PyTorch ONNX Export Debug

Systematic workflow for debugging PyTorch model → ONNX export failures.

## When to Activate

Activate this skill when:
- A PyTorch model fails to export to ONNX with `torch.onnx.export`
- The export completes but `onnxruntime` verification fails
- You see errors about `SymInt`, `torch.export`, or `dynamic_shapes`
- The exported ONNX graph is missing expected inputs or outputs
- You're on Windows and seeing `UnicodeEncodeError` with emoji characters
- A cached value or dict lookup fails during tracing

## Diagnostic Sequence

Follow these steps in order. Fixing an earlier issue often resolves later ones.

### Step 1: Isolate the Layer (Binary Search)
```python
# Comment out half the model forward(). If export succeeds, the bug is in the uncommented half.
# Repeat until you find the offending module.
```
- If the entire model fails: check import-level issues first (Step 2)
- If a specific module fails: inspect that module's patterns (Step 3)

### Step 2: Check the Export API
PyTorch 2.12+ changed the ONNX export internals to use `torch.export`:
- `torch.onnx.export` still works but `dynamic_axes` is converted to `dynamic_shapes` internally
- If you see `ValueError: Found the following conflicts between user-specified ranges and inferred ranges`: your `dynamic_axes` conflicts with traced shapes. Fix by either:
  - Use dummy inputs with the EXACT shape the model expects, and remove dynamic_axes
  - Or use `torch.export.Dim` API: `from torch.export import Dim; batch = Dim("batch", min=1, max=2)`
- **Never use `torch.onnx.ExportOptions`** — it doesn't exist in most PyTorch versions
- **Never use `torch.onnx.dynamo_export`** — it was experimental and removed/changed

### Step 3: Check Dict Caching Patterns
PyTorch tracing produces `SymInt` (symbolic integers) that are **deliberately unhashable**. Also, `torch.device` and `torch.dtype` objects become tensor proxies during tracing.

**Fix** all hash-based caches:
```python
# ❌ BAD — SymInt unhashable
key = (batch, seq_len, device)  # batch, seq_len are SymInt

# ✅ GOOD — convert to plain types
batch_int, seq_len_int = int(batch), int(seq_len)
key = (batch_int, seq_len_int, str(device), str(dtype))
```

**When in doubt**: `int()`, `str()`, `float()` any value before using it in a dict key during tracing.

### Step 4: Check Dynamic Control Flow
ONNX standard export (not dynamo) does NOT support dynamic control flow:
```python
# ❌ BAD — while loop depends on tensor values
while H * W > 2 * scale:
    H = (H + 1) // 2

# ✅ GOOD — static arithmetic for known sizes
import math
H_rope = H
W_rope = W
while (H_rope * W_rope) > 2 * scale:
    H_rope = (H_rope + 1) // 2
    W_rope = (W_rope + 1) // 2
```

For known sizes, precompute the loop result statically.

### Step 5: Check the Exported Graph
After exporting, inspect the ONNX graph:
```python
import onnx
model = onnx.load("model.onnx")
for inp in model.graph.input:
    print(f"Input: {inp.name}, shape: {[d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]}")
for out in model.graph.output:
    print(f"Output: {out.name}, shape: {[d.dim_param if d.dim_param else d.dim_value for d in out.type.tensor_type.shape.dim]}")
```

**Check**: If an input you expected is missing, the model doesn't actually use it (dead-code elimination). This is common for text embeddings in models that were trained without cross-attention.

### Step 6: Verify
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
for inp in session.get_inputs():
    print(f"ORT Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")
```

If verification OOMs on a consumer GPU (common for large attention matrices), try CPU or check with `onnx.checker.check_model()` first:
```python
onnx.checker.check_model(model)  # structural check, no GPU needed
```

### Step 7: Windows-Specific Fixes
On Windows, PyTorch's emoji error characters (✅ ❌) crash with `UnicodeEncodeError`:
```python
import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
```

## Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `TypeError: unhashable type: non-nested SymInt` | SymInt in dict key | Convert to `int()` before lookup |
| `AttributeError: module 'torch.onnx' has no attribute 'ExportOptions'` | Wrong API for PyTorch version | Use `torch.onnx.export()` with `dynamic_axes` |
| Missing ONNX inputs | Dead-code elimination optimized away unused inputs | Verify model actually uses them |
| `ValueError: conflict between user-specified ranges and inferred ranges` | `dynamic_axes`/`dynamic_shapes` mismatch | Use static dummy shapes matching reality |
| `GuardOnDataDependentSymNode` | Tensor `.max()` or `.min()` → data-dependent value | Precompute from static dimensions |
| `UnicodeEncodeError` with emoji | Windows cp1252 console | Reconfigure stdout/stderr to UTF-8 |

## Example

Debugging a UNet ONNX export that loads but fails at verification:

```
1. Run export → gets past model load but crashes at export
2. Error: "unhashable type: non-nested SymInt" → Fix dict caching (Step 3)
3. Re-run → crashes at different spot: "GuardOnDataDependentSymNode" → Fix tensor.max() with static dims
4. Re-run → exports but missing text_emb input → Dead-code elimination (Step 5)
5. Check model architecture → cross-attention not instantiated → expected behavior
```

## Pitfalls

- **Don't fix issues speculatively** — there's usually a chain of 3-7 bugs. Fix the first error, re-run, repeat.
- **Don't use `torch.onnx.dynamo_export`** — the stable `torch.onnx.export()` works better in PyTorch 2.12+
- **Don't assume all inputs are used** — ONNX's dead-code elimination is aggressive. Check the graph.
- **Don't try to verify on a consumer GPU** for large models — the attention MatMul may need 19+ GB. Use CPU verification or just `onnx.checker.check_model()`.
- **Don't use `.max()` or `.min()` on tensors during export** — they produce `GuardOnDataDependentSymNode` errors. Use static dimension arithmetic.
