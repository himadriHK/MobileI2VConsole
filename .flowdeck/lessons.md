## 2026-07-01 — FlowDeck orchestrator guard on Windows
**Severity:** medium
**Mistake:** Used Test-Path and Get-Item which were blocked by the orchestrator guard even though they are read-only operations on non-sensitive paths
**Lesson:** On Windows, the orchestrator guard restricts PowerShell cmdlets like Test-Path and Get-Item even for benign file checks. Use `read` (directory listing) or `bash ls` for file existence checks instead. fdx-ls is preferred but may not be in PATH on Windows.
## 2026-07-02 — PyTorch 2.x ONNX export with torch.export tracing
**Severity:** high
**Mistake:** Assumed dict-key caching with torch.device objects and Python ints would work during ONNX tracing. In PyTorch 2.12+, torch.onnx.export uses torch.export under the hood, which produces SymInt (symbolic integers) that are deliberately unhashable. Also torch.device objects become tensor proxies during tracing.
**Lesson:** When debugging PyTorch ONNX export failures in PyTorch 2.12+, always: (1) convert SymInt values to plain int() before using in dict keys, (2) use str(device) and str(dtype) instead of raw objects for caching keys, (3) replace while-loops that depend on tensor values with static integer arithmetic since ONNX standard export doesn't support dynamic control flow, (4) avoid dynamic_axes and use dynamic_shapes directly with torch.export.Dim for PyTorch 2.12+, (5) verify the model actually uses all its declared inputs — dead-code elimination may prune them.
## 2026-07-02 — Notebook editing: Fixed 5 discrepancies between Colab notebook and Python converter scripts
**Severity:** low
**Mistake:** Initially the SymInt fix to PositionGetter3D used arithmetic max_poses (t_int-1, h_int-1, w_int-1) which doesn't match the original int(poses[0].max()) behavior — but that's actually correct since positions are generated from torch.arange so max = size-1
**Lesson:** When editing Jupyter notebook JSON files, use `%%writefile` cells to sync Colab code with local scripts. Replacing the entire cell source array is more reliable than making surgical edits to the middle of large cells. Verify with Python json.load after every edit.
