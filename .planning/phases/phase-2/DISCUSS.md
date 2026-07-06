# Phase 2 — Fix & Verify Inference Pipeline

## Discuss Participants
- **User:** Requested verification of C# inference logic against exported ONNX models
- **@code-explorer:** Performed deep analysis of all 4 ONNX models vs C# code

## Source of Decisions
This discussion was conducted via Ask → @code-explorer analysis (2026-07-02).
The verification report identified 6 critical gaps between C# inference logic and exported ONNX models.

---

## D-01: IModelManager Must Expose Sessions

**Decision:** Extend `IModelManager` interface with a `GetSession(string modelName)` method so `InferenceOrchestrator` can access ONNX `InferenceSession` objects.

**Rationale:** Currently `_sessions` is `private ConcurrentDictionary` — orchestrator cannot run any inference.

**Impact:** All 4 pipeline steps are blocked until this is resolved.

---

## D-02: Latent Channels Must Be 128, Not 4

**Decision:** Change all latent channel constants from 4 to 128 throughout the C# pipeline.

**Rationale:** All 3 ONNX models (VAE encoder, UNet, Turbo-VAED) use 128-channel latents (LTX-Video architecture). The current C# code assumes 4-channel SD-style latents.

**Affected files:**
- `InferenceOrchestrator.cs` — `UnetLatentChannels` constant
- All pipeline step implementations

---

## D-03: VAE Encoder — Fix Input Size & Output Shape

**Decision:** VAE encoder should accept 256×256 input images (producing 32×32 latents) to match UNet expectations, not 512×512.

**Rationale:** The UNet expects 32×32 spatial latents. With 8× VAE compression, input must be 256×256. The C# code currently uses 512×512, producing 64×64 latents.

**Notes:** The ONNX model has dynamic H/W axes, so any multiple of 32 works. The actual production path (720p output) may need different dimensions.

---

## D-04: Qwen2 Encoder — Add Tokenizer & attention_mask

**Decision:** 
1. Implement Qwen2 byte-level BPE tokenizer in C# (or include ONNX-exported tokenizer)
2. Add `attention_mask` as a second input
3. Fix hidden size to 1024 (not 768 as stated in README)

**Rationale:** Currently `EncodeTextPromptAsync()` returns dummy data. The model expects `input_ids` + `attention_mask` both int64. No tokenizer exists in the project.

---

## D-05: UNet — Remove Text Embeddings, Fix 17-Frame Input

**Decision:**
1. Remove the `text_emb` input path — it's pruned from the ONNX graph by the JIT tracer
2. Accept `[1, 128, 17, 32, 32]` video latent as input (not single frame)
3. Fix spatial dimensions to 32×32 (not 64×64)
4. Implement proper timestep embedding computation

**Rationale:** The SanaBlock cross-attention layers accept `y` (text) but ignore it. The JIT tracer strips `text_emb` from the ONNX graph entirely. The UNet operates on 17-frame video latents at 32×32 spatial resolution with 128 channels.

---

## D-06: Turbo-VAED — Spatial Interpolation Required

**Decision:** Add spatial interpolation step between UNet output (32×32 latents) and VAED input (~23×40 for 720p). The VAED must be called with correct spatial dims for target output resolution.

**Rationale:** The UNet produces 32×32 spatial latents, but Turbo-VAED expects different spatial dimensions depending on target output resolution (e.g., 23×40 for 720p). A bilinear interpolation step is needed.

**Note:** The VAED output upscales by 32× in H/W and 8× in temporal dimension: `[1, 128, T, H, W] → [1, 3, T×8, H×32, W×32]`.

---

## D-07: NNAPI Acceleration Is Correct

**Decision:** Keep the existing NNAPI fallback pattern in `ModelManagerService.LoadModelAsync()`.

**Rationale:** Already correctly implemented with graceful CPU fallback.

---

## D-08: Test Coverage Required for All New Code

**Decision:** Every new inference method must have unit tests with 80%+ coverage. Mock the ONNX runtime for unit tests.

**Rationale:** No existing tests cover the inference pipeline. TDD approach — write failing test first.

---

## D-09: Phased Delivery

**Decision:** Deliver the fix in waves (interface → VAE → Qwen2 → UNet → VAED → integration) with verification gate after each wave.

**Rationale:** Each step depends on the previous. Independent waves minimize risk and allow incremental testing.
