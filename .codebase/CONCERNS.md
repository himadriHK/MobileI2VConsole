# Known Concerns

*Generated: 2026-07-02 | Root commit: 57ba4cb | 2 commits in history*

---

## Technical Debt

### 1. Entire ONNX Inference Pipeline Is Stubbed Out
The core `InferenceOrchestrator` (the app's raison d'être) returns **placeholder/dummy data** for all four ONNX model stages. Every pipeline step has a `// TODO: Implement actual ONNX inference — placeholder for now.` comment and returns zero-filled arrays instead of running real inference.

| Method | File | Line | Returns |
|--------|------|------|---------|
| `RunVaeEncoderAsync` | `src/MobileI2VConsole/Services/InferenceOrchestrator.cs` | 189 | `new float[1 * 4 * 64 * 64]` |
| `EncodeTextPromptAsync` | `src/MobileI2VConsole/Services/InferenceOrchestrator.cs` | 225 | `new float[1 * 16 * 1024]` |
| `RunUnetDenoisingAsync` | `src/MobileI2VConsole/Services/InferenceOrchestrator.cs` | 268 | `new float[17 * 4 * 64 * 64]` |
| `RunTurboVaedDecodeAsync` | `src/MobileI2VConsole/Services/InferenceOrchestrator.cs` | 302 | 17x `new float[3 * 1280 * 720]` |

**Impact:** The app compiles, navigates through all pages, and reports progress — but generates a black/corrupt 17-frame video. The "Generate" button is non-functional for actual I2V generation.

### 2. `GetSession()` Is `[Obsolete]` and Throws `NotImplementedException`
```csharp
// src/MobileI2VConsole/Services/InferenceOrchestrator.cs:380-387
[Obsolete("ONNX inference not yet wired — placeholder session accessor.")]
private static InferenceSession GetSession(string modelName)
{
    throw new NotImplementedException(
        $"ONNX inference requires an actual OrtSession for '{modelName}'.");
}
```
The `IModelManager` interface does not expose a `GetSession()` method, forcing the orchestrator to use this dead-end placeholder. Until this gap is closed, no ONNX inference can run.

### 3. Manifest Download Failure Silently Swallowed
In `ModelManagerService.cs:237-242`:
```csharp
catch
{
    // Manifest not available — proceed without SHA256 verification
    _manifest = null;
}
```
If the HuggingFace manifest URL fails or the network is unavailable, the download proceeds **without SHA256 verification**. Corrupted downloads won't be detected.

### 4. Bare `catch {}` Blocks Throughout C# Code
Multiple locations use empty or silent catch blocks that swallow all exceptions:

- `src/MobileI2VConsole/Services/ModelManagerService.cs:155` — NNAPI init failure swallowed silently
- `src/MobileI2VConsole/Services/ModelManagerService.cs:203` — Cache file delete failures swallowed
- `src/MobileI2VConsole/Services/FileService.cs:43` — Temp file cleanup best-effort
- `src/MobileI2VConsole/Services/InferenceOrchestrator.cs:122` — Model unload best-effort cleanup
- `src/MobileI2VConsole/Platforms/Android/AndroidVideoEncoder.cs:149,155` — MediaCodec stop/release errors swallowed

### 5. UTF-8 Console Workaround Duplicated in Python Code
The same UTF-8 reconfigure block (for Windows cp1252 emoji crash) appears in both:
- `scripts/convert/common/model_utils.py:25-34`
- `scripts/convert/../MobileI2V_ONNX_Converter.ipynb` (inline duplicated cell)

This should be a shared utility.

### 6. Notebook Duplicates Entire `model_utils.py` Inline
The Colab notebook `scripts/notebooks/MobileI2V_ONNX_Converter.ipynb` writes out a complete copy of `model_utils.py` via `%%writefile` magic. This duplicates ~150 lines of code that must be kept in sync with the canonical `common/model_utils.py`.

### 7. `convert_mobilei2v_unet.py` Monkey-Patches `Attention.forward`
The UNet converter temporarily replaces `Attention.forward` with a patched version using `scaled_dot_product_attention` to avoid OOM on 4 GB GPUs (`scripts/convert/convert_mobilei2v_unet.py:195-218`). While restored in a `try/finally`, this is fragile — if an exception occurs in the export wrapper code path that doesn't go through the finally block (e.g., a segfault), the monkey-patch could persist.

### 8. `verify_onnx` Has Dead Parameters
In `scripts/convert/common/model_utils.py:315-316`:
```python
rtol: float = 1e-3,
atol: float = 1e-3,
```
These tolerance parameters are declared but **never used** in the function body. The docstring also admits they're unused: "not used yet".

---

## Known Bugs / Issues

### 1. TurboVAEDDecoder3d `inject_noise` Tuple Length (FIXED — Regression Test Added)
**Root cause:** The `inject_noise` default tuple had 4 elements but needed 5 (index 0 for mid block + 4 for up blocks). This caused an `IndexError` on instantiation.

**Fix:** Changed from `(False,False,False,False)` to `(False,False,False,False,False)` in both the constructor signature (`turbo_vaed_model.py:585`) and the factory `setdefault` (`turbo_vaed_model.py:783`).

**Regression test:** Added at `scripts/convert/tests/test_turbo_vaed_bug.py`.

**Source:** `.codebase/FAILURES.json` entry F-1.

### 2. UNet `text_emb` Input Pruned by JIT Tracer During ONNX Export
The MobileI2V UNet's `text_emb` input is included in the ONNX graph's `input_names` and `dynamic_axes`, but the JIT tracer prunes it because `SanaBlock_cross` forwards accept `y` as a parameter but **ignore it** (`scripts/convert/convert_mobilei2v_unet.py:163-164` and `238`). The exported ONNX model will have only 2 inputs (`latent` and `timestep`) instead of 3, which may cause problems at runtime if the app tries to feed a `text_emb` tensor.

### 3. ONNX Runtime Verification OOM for UNet
The `verify_onnx` call for the UNet model is known to OOM on 4 GB GPUs due to the 17-frame attention matrix (`scripts/convert/convert_mobilei2v_unet.py:248-260`). The error is caught and logged as a warning, but verification is skipped. The model may have runtime issues that only appear on-device.

### 4. `FileService` Uses Real Filesystem in Tests
`tests/MobileI2VConsole.Tests/Services/FileServiceTests.cs:10` creates a real temp directory for testing, rather than using a mock. The `FileService` constructor also uses `Environment.SpecialFolder.Personal` which behaves differently on Android vs the test runner. `InitializeAsync_CreatesDirectories` test is non-deterministic.

### 5. AndroidVideoEncoder Uses Deprecated `GetInputBuffers`/`GetOutputBuffers`
The Android `MediaCodec` API methods `GetInputBuffers()` and `GetOutputBuffers()` (`AndroidVideoEncoder.cs:67-68`) are deprecated in API 29+ (Android 10+). They have been replaced by the `GetInputBuffer(int)` / `GetOutputBuffer(int)` and `OnInputBufferAvailable` / `OnOutputBufferAvailable` callbacks. The current approach may cause compatibility issues on newer Android versions.

### 6. `GenerationViewModel.OnRequestChanged` Fire-and-Forget
`src/MobileI2VConsole/ViewModels/GenerationViewModel.cs:56`:
```csharp
_ = StartGenerationAsync();
```
This is a fire-and-forget async call. If `StartGenerationAsync` throws an exception before the method body's `try/catch` (e.g., during `JsonSerializer.Deserialize` on the UI thread at line 71), the exception will crash the app because there is no exception handler wrapping the `_ = ...` pattern.

---

## Security Concerns

### 1. `weights_only=False` Fallback in UNet Checkpoint Loader
`scripts/convert/convert_mobilei2v_unet.py:106`:
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```
If `weights_only=True` fails due to non-tensor metadata, the code falls back to `weights_only=False`, which executes arbitrary pickle code. The code has a warning comment but still does it — a supply-chain risk if the HuggingFace checkpoint is ever compromised.

### 2. Silent Catching of All Exceptions in Manifest Download
`src/MobileI2VConsole/Services/ModelManagerService.cs:238` and download retry `:128` catch every exception type. This includes `OperationCanceledException` in the retry loop (which should propagate). The manifest catch means a tampered manifest or MITM attack on the manifest URL wouldn't be detected — models could be downloaded without SHA256 verification.

### 3. No Output Path Validation
In `GenerationViewModel.cs:93-94`, the output path is constructed from `DateTime.Now` and combined with the output directory:
```csharp
var outputPath = Path.Combine(_fileService.GetOutputDirectory(),
    $"video_{DateTime.Now:yyyyMMddHHmmss}.mp4");
```
While not directly injectable (no user input in the filename), there is no path traversal protection or validation if `GetOutputDirectory()` is ever modified.

---

## Performance Concerns

### 1. All 4 ONNX Models Loaded Simultaneously (~1.5 GB)
`InferenceOrchestrator.cs:58` loads all four models at once:
```csharp
var modelsToLoad = new[] { "vae_encoder", "qwen2_encoder", "mobilei2v_unet", "turbo_vaed" };
```
The Qwen2-0.5B model alone is ~900 MB. On a device with 2 GB RAM, loading all four simultaneously will likely cause an `OutOfMemoryException` or GC thrashing. Models should be loaded on-demand and unloaded after use.

### 2. Unnecessary Model Unload/Reload on Every Generation
`InferenceOrchestrator.cs:118-123` unloads all models in a `finally` block after every generation, including models that may already have been loaded. If the user runs generation back-to-back, every model must be re-downloaded from disk and re-created.

### 3. Inefficient RGBA→NV12 Conversion Loop
`AndroidVideoEncoder.ConvertRgbaToNv12` (`AndroidVideoEncoder.cs:171-210`) uses a nested `for` loop over every pixel (1280×720 = 921,600 px × 17 frames). This is done in C# with no SIMD or hardware acceleration. Each iteration performs multiple integer arithmetic operations. For 17 frames, this is ~15.6 million pixel operations.

### 4. `OrtEnv` Recreated Potentially on Each Load
`ModelManagerService.cs:150-151`:
```csharp
if (_ortEnv == null || _ortEnv.IsInvalid)
    _ortEnv = OrtEnv.Instance();
```
The `OrtEnv` is a singleton in onnxruntime, but `OrtEnv.Instance()` may return an existing instance. The null/IsInvalid check is insufficient — if the environment was disposed externally, this could create a new one.

### 5. HTTP Client Has Fixed 30-Minute Timeout
`ModelManagerService.cs:36`:
```csharp
_httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
```
No per-request cancellation token integration beyond the `CancellationToken` parameter. If a download hangs, the operation will block for up to 30 minutes.

### 6. `ExpandableSegments` Workaround for CUDA Memory Fragmentation
The Colab notebook at `scripts/notebooks/MobileI2V_ONNX_Converter.ipynb` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid OOM on 4 GB GPUs. This is a workaround for PyTorch memory fragmentation, not a fix.

---

## Dependency Risks

### 1. Unpinned Python Dependencies
`scripts/convert/requirements.txt` uses `>=` version ranges for all dependencies (e.g., `torch>=2.1.0`, `transformers>=4.44.0`, `onnx>=1.15.0`). This means builds can break when newer incompatible versions are published. There is no lockfile (`requirements.lock`, `poetry.lock`, etc.).

### 2. Net10.0-android Target Framework Is Prerelease
`src/MobileI2VConsole/MobileI2VConsole.csproj:4` targets `net10.0-android`. .NET 10 is a preview release (not yet stable as of mid-2026). API surface may change before the stable release, and some MAUI workloads may have bugs.

### 3. PyTorch 2.12+ ONNX Exporter Changes
The codebase handles the PyTorch 2.12+ `dynamo=True` default change with a version check (`_torch_version_tuple`) and `dynamo=False` workaround (`model_utils.py:268-289`). This is fragile — if PyTorch changes the default again or deprecates `dynamo=False`, the export pipeline breaks.

### 4. `onnxruntime-gpu` and `onnxruntime` Conflict
The `requirements.txt` warns users not to install both `onnxruntime` and `onnxruntime-gpu`, but provides no automated conflict detection.

### 5. `onnxruntime.transformers.optimizer` Has Hardcoded `model_type="bert"`
In `scripts/convert/common/ort_config.py:51`:
```python
opt = t_optimizer.optimize_model(
    str(onnx_path),
    model_type="bert",  # generic; works for non-BERT models too
    ...
)
```
The optimizer is called with `model_type="bert"` even for non-BERT models (VAE, UNet, Turbo-VAED). While the comment says it "works for non-BERT models too", this is undocumented behavior and could produce suboptimal optimization.

### 6. `CommunityToolkit.Maui.MediaElement` v10.0.0 at Risk
`MobileI2VConsole.csproj:42` pins `CommunityToolkit.Maui.MediaElement` to version `10.0.0`. This package has historically had versioning issues (not all versions align with MAUI releases). The `false` parameter in `.UseMauiCommunityToolkitMediaElement(false)` at `MauiProgram.cs:17` suggests MediaElement features are partially disabled.

### 7. NuGet Dependencies on .NET 10 Preview Packages
All NuGet packages (`Microsoft.Maui.Controls 10.0.80`, `CommunityToolkit.*`, `SkiaSharp 4.148.0`, `Microsoft.ML.OnnxRuntime 1.27.0`) are forward-compatible with .NET 10. However, these are early-adopter versions.

---

## Testing Gaps

### 1. No Tests for InferenceOrchestrator
The core pipeline orchestrator (`InferenceOrchestrator`) has **zero unit tests**. Given that all pipeline methods return dummy data, tests would have caught this early. Critical areas untested:
- Pipeline step ordering
- Progress reporting boundary values
- Cancellation propagation
- Error handling (model load failure, partial failures)

### 2. No Tests for AndroidVideoEncoder
The Android-specific video encoder (`AndroidVideoEncoder`) has no tests. The `ConvertRgbaToNv12` color conversion is a prime candidate for unit testing with known RGBA input and expected NV12 output.

### 3. No Tests for ResultViewModel
The `ResultViewModel` (`src/MobileI2VConsole/ViewModels/ResultViewModel.cs`) has no tests. Share/save/view functionality is untested.

### 4. Python Tests: Only 1 Regression Test
The entire Python test suite consists of a single 25-line regression test (`scripts/convert/tests/test_turbo_vaed_bug.py`). There are no tests for:
- `export_onnx` verification
- `verify_onnx` edge cases
- `_patch_neg_transpose` functionality
- Converter wrappers (VAEEncoderWrapper, Qwen2EncoderWrapper, etc.)

### 5. Missing Coverage Configuration
While `coverlet.collector` is included in the test project (`tests/MobileI2VConsole.Tests.csproj:17`), there is no coverage threshold configuration or `lcov`/`cobertura` report export setup.

### 6. FileServiceTests Use Real I/O
`FileServiceTests.cs` calls the real `FileService` constructor (which uses `Environment.SpecialFolder.Personal`) instead of injecting a mock. The `InitializeAsync_CreatesDirectories` test is fragile because it depends on the test host's file system.

### 7. No Integration or E2E Tests
There are no integration tests that verify the end-to-end pipeline (even with stub data) or UI E2E tests.

---

## Process Issues

### 1. ONNX Model Binaries Not in `.gitignore`
The git status shows untracked ONNX binary files:
```
?? scripts/convert/mobilei2v_unet.onnx
?? src/MobileI2VConsole/Resources/Raw/mobilei2v_unet.onnx
?? src/MobileI2VConsole/Resources/Raw/qwen2_encoder.onnx
?? src/MobileI2VConsole/Resources/Raw/qwen2_encoder.onnx.data
?? src/MobileI2VConsole/Resources/Raw/turbo_vaed.onnx
?? src/MobileI2VConsole/Resources/Raw/vae_encoder.onnx
```
These should be added to `.gitignore` and generated via a build step or download script instead.

### 2. Debug Logging Commented Out
`src/MobileI2VConsole/MauiProgram.cs:51`:
```csharp
// Debug logging omitted - AddDebug requires Microsoft.Extensions.Logging.Debug package
```
The `Microsoft.Extensions.Logging.Debug` package is not included in the project file, so debug logging is completely absent. If a runtime crash occurs, there are no logs to diagnose it.

### 3. No CI/CD Pipeline
There is no visible CI configuration (GitHub Actions, Azure DevOps, etc.) in the repository. The project cannot be built or tested automatically.

### 4. README Documents Full Functionality That Doesn't Exist
The `README.md` describes the full inference pipeline with detailed architecture diagrams and step-by-step explanations, but the actual implementation returns placeholder data. A user reading the README would expect a working app.

### 5. `.gitignore` Needs Review
The current `.gitignore` may not exclude build artifacts for the `scripts/convert/model_cache/` directory or the `scripts/convert/*.onnx` files.

### 6. Only 2 Commits in Git History
The repository has only 2 commits (`83cfc92` initial commit + `57ba4cb` new notebook). This is very early stage — there is no commit history to understand design decisions or track regressions.

---

## Recently Fixed Issues (for context)

### F-1: TurboVAEDDecoder3d `inject_noise` IndexError
- **Type:** Bug fix
- **Root cause:** `inject_noise` default tuple had 4 elements but needed 5 (index 0 for mid block + 4 up blocks)
- **Fix:** Changed default from `(False,False,False,False)` to `(False,False,False,False,False)` in both constructor (`turbo_vaed_model.py:585`) and factory setdefault (`turbo_vaed_model.py:783`)
- **Regression test:** `scripts/convert/tests/test_turbo_vaed_bug.py`
- **Recurrence risk:** Low — regression test covers instantiation

### Learned Policies (from `.codebase/POLICIES.json`)

| Policy | Trigger | Rule |
|--------|---------|------|
| `pol-001` | ONNX export completion | Always run `verify_onnx()`; if OOM, at minimum run `onnx.checker.check_model()` |
| `pol-002` | Before ONNX export | Audit which forward() parameters are actually consumed; JIT tracer may prune unused inputs |

---

## Summary Risk Assessment

| Category | Severity | Count |
|----------|----------|-------|
| 🔴 Critical — blocks functionality | Inferred pipeline returns dummy data | 1 |
| 🟠 High — production blocker | No ONNX inference, missing DI for sessions | 3 |
| 🟡 Medium — needs attention | Monkey-patching, deprecations, security warnings | 10 |
| 🔵 Low — improvement | Dead params, missing tests, duplicated code | 8 |

The most critical issue is that `InferenceOrchestrator` returns placeholder data for all four ONNX model stages, making the app non-functional for its primary purpose (image-to-video generation). Until `IModelManager` exposes an `GetSession()` or equivalent method and the four `Run*Async` methods execute real ONNX inference, the app is essentially a UI shell.
