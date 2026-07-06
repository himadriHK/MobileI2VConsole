# Phase 2 Plan — Fix & Verify Inference Pipeline

## Overview

Rewrite the C# ONNX inference pipeline from placeholder/dummy implementations to actual ONNX Runtime inference calls, matching the exact shapes, channels, and data types of the exported ONNX models in `Resources/Raw/`.

**Status:** Draft — awaiting confirmation
**Total Tasks:** 18 across 6 waves
**Estimated impact:** 6 source files modified, 3 test files created/modified

---

## Wave 1: Foundation — Session Wiring

### Task 1.1 — Extend IModelManager to Expose Sessions

**D-01** | **Files:** `IModelManager.cs`, `ModelManagerService.cs`, `InferenceOrchestrator.cs`

Add `InferenceSession? GetSession(string modelName)` to the interface and implementation.

```csharp
// IModelManager — new method
InferenceSession? GetSession(string modelName);
```

In `ModelManagerService`, expose the existing `_sessions` dictionary lookup.
In `InferenceOrchestrator`, replace `[Obsolete] GetSession()` with calls to `_modelManager.GetSession()`.

**Verification:** New unit test on `ModelManagerServiceTests` — verifies `GetSession` returns session after `LoadModelAsync`, null before.

---

### Task 1.2 — Add InferenceSession using to Orchestrator

**D-01** | **Files:** `InferenceOrchestrator.cs` (top of file)

```csharp
using Microsoft.ML.OnnxRuntime;
```

Already present. Verify.

---

### Task 1.3 — Write Tests for Session Wiring

**D-01, D-08** | **Files:** `tests/MobileI2VConsole.Tests/Services/ModelManagerServiceTests.cs`

Add test class `GetSessionTests`:
- `GetSession_ReturnsNull_BeforeLoad`
- `GetSession_ReturnsSession_AfterLoad`
- `GetSession_ReturnsNull_AfterUnload`

---

## Wave 2: VAE Encoder Pipeline Step

### Task 2.1 — Fix Latent Channel Constant

**D-02** | **Files:** `InferenceOrchestrator.cs`

Change `UnetLatentChannels = 4` to `UnetLatentChannels = 128`.

Change all derived constants and array allocations that depend on this value.

**Verification:** Unit test verifies constant value.

---

### Task 2.2 — Fix VAE Encoder Input Size

**D-03** | **Files:** `InferenceOrchestrator.cs`

Change `VaeInputSize = 512` to `VaeInputSize = 256`.

This produces `LatentH = LatentW = 32` (256/8) which matches UNet expectations.

**Rationale:** 8× VAE compression. 256×256 input → 32×32 latent. UNet expects 32×32 spatial.

**Verification:** Constant value test.

---

### Task 2.3 — Implement Real VAE Encoder ONNX Inference

**D-01, D-02, D-03, D-08** | **Files:** `InferenceOrchestrator.cs`

Replace `TODO` placeholder in `RunVaeEncoderAsync()` with:

```csharp
private async Task<float[]> RunVaeEncoderAsync(float[] preprocessedImage, CancellationToken ct)
{
    return await Task.Run(() =>
    {
        var session = _modelManager.GetSession("vae_encoder")
            ?? throw new InvalidOperationException("vae_encoder not loaded");

        using var inputOrt = OrtValue.CreateTensorValueFromMemory(
            OrtAllocator.DefaultInstance,
            preprocessedImage,
            new long[] { 1, 3, VaeInputSize, VaeInputSize });

        var inputs = new Dictionary<string, OrtValue> { ["pixel_values"] = inputOrt };
        var outputs = new List<string> { "latent" };

        using var result = session.Run(new RunOptions(), inputs, outputs);
        var latent = result[0].GetTensorDataAsSpan<float>().ToArray();

        return latent;  // shape: [1, 128, 32, 32]
    }, ct);
}
```

**Verification:** 
- Unit test with mocked session verifies correct input names and shapes
- Integration test with actual ONNX model produces correct output shape

---

### Task 2.4 — Write VAE Encoder Tests

**D-08** | **Files:** `tests/MobileI2VConsole.Tests/Services/InferenceOrchestratorTests.cs`

New test class:
- `RunVaeEncoderAsync_Throws_WhenModelNotLoaded`
- `RunVaeEncoderAsync_ReturnsCorrectShape` (mock session)
- `LoadAndPreprocessImage_ResizesTo256x256`
- `LoadAndPreprocessImage_NormalizesToMinus1To1`

---

## Wave 3: Qwen2 Text Encoder

### Task 3.1 — Port Qwen2 Tokenizer

**D-04** | **Files:** New file `Services/Qwen2Tokenizer.cs`

Implement Qwen2 byte-level BPE tokenizer in C#. Qwen2-0.5B uses the same tokenizer as Qwen2 (based on tiktoken/BPE).

Approach options (choose one):
- **Option A:** Export the Qwen2 tokenizer to ONNX and include it as a separate model
- **Option B:** Port the tokenizer manually (vocab.json + merges.txt → C# BPE implementation)
- **Option C:** Use a NuGet package for HuggingFace tokenizers (if available)

**Recommended:** Option A or C (simpler than manual BPE port).

**Verification:** Unit test tokenizes known strings and matches expected token IDs from Python.

---

### Task 3.2 — Add attention_mask Input

**D-04** | **Files:** `InferenceOrchestrator.cs`

Modify `EncodeTextPromptAsync()` to:
1. Tokenize prompt → `input_ids` (int64)
2. Create `attention_mask` (int64) — 1 for real tokens, 0 for padding
3. Create both OrtValue tensors
4. Run Qwen2 ONNX session with both inputs
5. Extract `last_hidden_state` from output

```csharp
var inputs = new Dictionary<string, OrtValue>
{
    ["input_ids"] = inputIdsOrt,
    ["attention_mask"] = attentionMaskOrt,
};
```

**Verification:** Unit test verifies both inputs are passed to session.Run().

---

### Task 3.3 — Fix Hidden Size to 1024

**D-04** | **Files:** `InferenceOrchestrator.cs` (EncodeTextPromptAsync and constants)

Change dummy placeholder from `[1, 1, 1024]` to proper output. Verify actual hidden size is 1024.

Note: Qwen2-0.5B hidden size is 1024, not 768 (README is inaccurate — update README too).

---

### Task 3.4 — Write Qwen2 Tests

**D-08** | **Files:** `tests/MobileI2VConsole.Tests/Services/Qwen2TokenizerTests.cs`

- `Tokenize_EmptyString_ReturnsSingleToken`
- `Tokenize_KnownString_MatchesExpectedIds`
- `Tokenize_TruncatesAtMaxLength`
- `EncodeTextPromptAsync_WithEmptyPrompt_ReturnsZeroEmbeddings`
- `EncodeTextPromptAsync_PassesAttentionMask`

---

## Wave 4: UNet Denoising

### Task 4.1 — Restructure UNet Input to 17-Frame Video

**D-05** | **Files:** `InferenceOrchestrator.cs`

Change `RunUnetDenoisingAsync()` input/output:

**Before:**
```
Input:  [1, 4, 64, 64]   — single frame latent
Steps:  2-step diffusion loop expanding latent
Output: [17, 4, 64, 64]  — expanded after denoising
```

**After:**
```
Input:  [1, 128, 17, 32, 32]  — full 17-frame video latent
Steps:  Add noise → run UNet → output denoised latent
Output: [1, 128, 17, 32, 32]  — same shape as input (denoised)
```

Key changes:
- VAE encoder output `[1, 128, 32, 32]` must be expanded to `[1, 128, 17, 32, 32]` (repeat or add noise per frame)
- Remove text embedding concatenation (pruned from ONNX graph)
- Compute timestep as int64 tensor `[1]`
- Run a single session.Run() call per diffusion step

---

### Task 4.2 — Remove Text Embedding Path from UNet

**D-05** | **Files:** `InferenceOrchestrator.cs`

Remove `textEmbeddings` parameter from `RunUnetDenoisingAsync()` or mark it unused. The ONNX graph doesn't consume `text_emb` — it's pruned.

**Verification:** ONNX graph inspection confirms `text_emb` not in input list.

---

### Task 4.3 — Fix Spatial Dimensions to 32×32

**D-05** | **Files:** `InferenceOrchestrator.cs`

Ensure `LatentH = LatentW = 32` (already done in Task 2.2). All UNet-related array allocations use 32×32.

**Verification:** Shape assertion in unit test.

---

### Task 4.4 — Write UNet Denoising Tests

**D-08** | **Files:** `tests/MobileI2VConsole.Tests/Services/InferenceOrchestratorTests.cs`

- `RunUnetDenoisingAsync_Throws_WhenModelNotLoaded`
- `RunUnetDenoisingAsync_InputShape_Is128x17x32x32`
- `RunUnetDenoisingAsync_TextEmbeddingsNotPassedToSession`
- `RunUnetDenoisingAsync_OutputShape_SameAsInput`

---

## Wave 5: Turbo-VAED Decoder

### Task 5.1 — Add Spatial Interpolation Step

**D-06** | **Files:** `InferenceOrchestrator.cs` or new `Services/SpatialInterpolator.cs`

Add a spatial interpolation function between UNet output and VAED input:

```csharp
// UNet produces [1, 128, 17, 32, 32] — 32×32 spatial
// VAED expects [1, 128, 17, H_vaed, W_vaed] where H_vaed,W_vaed determined by output resolution
// Bilinear interpolation from 32×32 to target spatial dims
float[] InterpolateLatents(float[] unetOutput, int targetH, int targetW);
```

For 720p output:
- VAED input spatial: H=23, W=40 (produces 736×1280 output)
- Need interpolation: 32→23 in H, 32→40 in W

**Verification:** Unit test verifies output shape and approximate value preservation.

---

### Task 5.2 — Implement Real VAED ONNX Inference

**D-01, D-02, D-06, D-08** | **Files:** `InferenceOrchestrator.cs`

Replace `TODO` placeholder in `RunTurboVaedDecodeAsync()`:

```csharp
private async Task<float[][]> RunTurboVaedDecodeAsync(
    float[] latentFrames, int targetH, int targetW,
    IProgress<GenerationProgress>? progress, CancellationToken ct)
{
    return await Task.Run(() =>
    {
        var session = _modelManager.GetSession("turbo_vaed");
        // Interpolate spatial dims
        var interpolated = InterpolateLatents(latentFrames, vaedH, vaedW);
        // Create input tensor [1, 128, 17, vaedH, vaedW]
        using var inputOrt = OrtValue.CreateTensorValueFromMemory(...);
        var inputs = new Dictionary<string, OrtValue> { ["latent"] = inputOrt };
        var outputs = new List<string> { "frames" };
        using var result = session.Run(new RunOptions(), inputs, outputs);
        // Output: [1, 3, 136, H*32, W*32]
        // Split into 17 individual frame arrays
        var frames = ExtractFrames(result[0], 17, 3, targetW, targetH);
        return frames;
    }, ct);
}
```

**Verification:** Unit test with mocked session.

---

### Task 5.3 — Write VAED Decode Tests

**D-08** | **Files:** `tests/MobileI2VConsole.Tests/Services/InferenceOrchestratorTests.cs`

- `RunTurboVaedDecodeAsync_Throws_WhenModelNotLoaded`
- `RunTurboVaedDecodeAsync_Returns17Frames`
- `InterpolateLatents_ChangesSpatialDimensions`
- `RunTurboVaedDecodeAsync_CorrectInputShape`

---

## Wave 6: Integration & Verification

### Task 6.1 — Wire Full Pipeline End-to-End

**D-01 to D-09** | **Files:** `InferenceOrchestrator.cs`, `GenerationViewModel.cs`

Update `GenerateVideoAsync()` to chain the real pipeline steps:

```
1. Load all 4 models (existing — works)
2. VAE encode: image → latent [1,128,32,32] (Task 2.3)
3. Expand: single latent → 17-frame [1,128,17,32,32] (add noise per frame)
4. Qwen2 encode: text → embeddings [1,seq_len,1024] (Task 3.2)
   Note: text_emb is NOT passed to UNet (pruned from graph — Task 4.2)
5. UNet denoise: [1,128,17,32,32] → denoised [1,128,17,32,32] (Task 4.1)
6. Interpolate spatial: 32² → target dims for VAED (Task 5.1)
7. VAED decode: → 17 RGBA frames (Task 5.2)
8. Convert floats → byte RGBA (existing — works)
```

**Verification:** Full integration test with mocked ONNX sessions.

---

### Task 6.2 — Integration Test

**D-08, D-09** | **Files:** `tests/MobileI2VConsole.Tests/Services/InferenceOrchestratorTests.cs`

- `GenerateVideoAsync_FullPipeline_Returns17Frames`
- `GenerateVideoAsync_ReportsProgressThroughAllSteps`
- `GenerateVideoAsync_UnloadsModelsOnError`
- `GenerateVideoAsync_CancellationStopsMidPipeline`

---

### Task 6.3 — Verification Against ONNX Models

**D-09** | **Files:** Manual verification step

Run actual ONNX inference with known input image and prompt:
1. Load VAE encoder → encode image → verify output shape [1,128,32,32]
2. Load Qwen2 encoder → encode text → verify output shape [1,seq_len,1024]
3. Load UNet → denoise → verify output shape matches input
4. Load VAED → decode → verify 17 RGBA frames at correct resolution
5. Full pipeline → produce MP4 video → verify playback

---

## Wave Dependency Graph

```
Wave 1 (Foundation)
  └── Task 1.1 ─── Task 1.3
  └── Task 1.2 (trivial, parallel)

Wave 2 (VAE) ── depends on Wave 1
  └── Task 2.1 ─── Task 2.2 (trivial, parallel)
  └── Task 2.3 ─── Task 2.4

Wave 3 (Qwen2) ── depends on Wave 1 only (parallel with Wave 2)
  └── Task 3.1 ─── Task 3.2 ─── Task 3.4
  └── Task 3.3 (trivial, parallel with 3.2)

Wave 4 (UNet) ── depends on Wave 2 (needs correct VAE output shape)
  └── Task 4.1 ─── Task 4.2 (parallel)
  └── Task 4.3 (trivial, parallel)
  └── Task 4.4

Wave 5 (VAED) ── depends on Wave 2, 4 (needs correct channels + spatial)
  └── Task 5.1 ─── Task 5.2 ─── Task 5.3

Wave 6 (Integration) ── depends on all previous waves
  └── Task 6.1 ─── Task 6.2 ─── Task 6.3
```

---

## Files Modified

| File | Wave | Change |
|------|------|--------|
| `Services/IModelManager.cs` | 1 | Add `GetSession()` method |
| `Services/ModelManagerService.cs` | 1 | Implement `GetSession()`, expose `_sessions` |
| `Services/InferenceOrchestrator.cs` | 1-6 | Replace all 4 pipeline step placeholders, fix constants |
| `Services/Qwen2Tokenizer.cs` | 3 | **New file** — BPE tokenizer for Qwen2 |
| `Services/SpatialInterpolator.cs` | 5 | **New file** — bilinear interpolation for latents |
| `Tests/.../ModelManagerServiceTests.cs` | 1 | Add session wiring tests |
| `Tests/.../Qwen2TokenizerTests.cs` | 3 | **New file** — tokenizer unit tests |
| `Tests/.../InferenceOrchestratorTests.cs` | 2-6 | **New file** — full inference tests |

---

## Risk Assessment

| Risk | Mitigation | Wave |
|------|-----------|------|
| Qwen2 tokenizer complexity is underestimated | Option C (NuGet package) preferred; fallback to Option A | 3 |
| Spatial interpolation adds latency | Keep as simple bilinear; optimize if needed | 5 |
| VAED output shape depends on export parameters | Verify actual ONNX model input/output shapes before implementing | 5 |
| Integration test needs actual ONNX models | Mock ONNX runtime for unit tests; manual integration test | 6 |
| Pipeline latency on mobile (4 models) | NNAPI acceleration already configured; monitor in verification | 6 |

---

## Verification Gates

After each wave, run:
1. `dotnet build` — must compile without errors
2. `dotnet test` — all tests must pass
3. Code review of new/changed files

After Wave 6 (Integration):
4. Manual verification with actual device/emulator
5. Verify output video plays correctly
