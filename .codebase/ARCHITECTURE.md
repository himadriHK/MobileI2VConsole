# Architecture

## Overview

MobileI2V Console is an Android .NET MAUI application that performs **on-device image-to-video generation**. Users select an image and optionally describe motion with text; the app runs four ONNX models sequentially via ONNX Runtime to generate 17 frames, then encodes them into a 720p H.264 MP4 video using Android MediaCodec. All inference runs locally — no cloud API calls.

The project has two distinct subsystems: a **C# .NET MAUI Android app** (the UI + inference runtime) and a **Python model conversion pipeline** (PyTorch → ONNX export for all four models).

**Architectural style:** MVVM (Model-View-ViewModel) with Dependency Injection for the mobile app; sequential pipeline stages for both inference and model conversion.

---

## System Design

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MobileI2V Console (.NET MAUI)                    │
│                                                                     │
│  ┌────────────┐    ┌────────────────┐    ┌──────────────┐          │
│  │  HomePage   │───▶│ GenerationPage │───▶│  ResultPage   │          │
│  │  (input)    │    │  (progress)    │    │  (preview)   │          │
│  └──────┬──────┘    └───────┬────────┘    └──────┬───────┘          │
│         │                   │                     │                  │
│  ┌──────▼───────────────────▼─────────────────────▼───────┐         │
│  │                   ViewModels (MVVM)                      │         │
│  │  HomeViewModel  |  GenerationViewModel  |  ResultViewModel │         │
│  └──────▲───────────────────▲─────────────────────▲───────┘         │
│         │                   │                     │                  │
│  ┌──────▼───────────────────▼─────────────────────▼───────┐         │
│  │                Services (DI Container)                   │         │
│  │                                                          │         │
│  │  ┌────────────┐  ┌──────────┐  ┌────────────────────┐  │         │
│  │  │ModelManager│  │FileService│  │InferenceOrchestrator│  │         │
│  │  │(download + │  │(paths +  │  │(pipeline runner)   │  │         │
│  │  │ ONNX sess.)│  │cleanup)  │  └────────┬───────────┘  │         │
│  │  └────────────┘  └──────────┘           │              │         │
│  │  ┌────────────────┐  ┌────────────────┐ │              │         │
│  │  │MediaPickerSvc  │  │VideoEncoder    │◀┘              │         │
│  │  │(gallery/camera)│  │(H.264 → MP4)   │                │         │
│  │  └────────────────┘  └────────────────┘                │         │
│  └─────────────────────────────────────────────────────────┘         │
│                           │                                          │
│  ┌────────────────────────▼─────────────────────────────────────────┐│
│  │                 ONNX Runtime Inference Pipeline                    ││
│  │                                                                    ││
│  │  Step 1          Step 2          Steps 3-4          Step 5        ││
│  │  ┌──────────┐   ┌────────────┐  ┌──────────────┐  ┌───────────┐  ││
│  │  │VAE Enc. │   │Qwen2-0.5B  │  │MobileI2V UNet│  │Turbo-VAED│  ││
│  │  │720p→lat │   │txt→emb     │  │(2-step diff.)│  │lat→frame  │  ││
│  │  │~50 MB   │   │~900 MB     │  │~540 MB       │  │~50 MB     │  ││
│  │  └────┬────┘   └─────┬──────┘  └──────┬───────┘  └─────┬─────┘  ││
│  │       │              │                │                │         ││
│  │       └──────┬───────┘                │          Step 6 │         ││
│  │              ▼                        ▼                ▼         ││
│  │          latent [1,4,64,64]    17 latent frames  17 RGBA frames  ││
│  │                                                     │            ││
│  │                                          ┌──────────▼─────────┐ ││
│  │                                          │Android MediaCodec  │ ││
│  │                                          │H.264 → MP4 @17FPS │ ││
│  │                                          └────────────────────┘ ││
│  └──────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│             Python Model Conversion Pipeline (scripts/convert/)       │
│                                                                      │
│  convert_all.py (orchestrator)                                       │
│  ├── Phase 1: convert_vae_encoder.py    → vae_encoder.onnx          │
│  ├── Phase 2: convert_qwen2_encoder.py  → qwen2_encoder.onnx        │
│  ├── Phase 3: convert_mobilei2v_unet.py → mobilei2v_unet.onnx       │
│  ├── Phase 4: convert_turbo_vaed.py     → turbo_vaed.onnx           │
│  └── Post:    compute_ort_export() + build_manifest() → .ort + JSON  │
│                                                                      │
│  Common utilities: model_utils.py, ort_config.py                     │
│  Vendored models:  models/mobiledit.py, models/turbo_vaed_model.py  │
└──────────────────────────────────────────────────────────────────────┘
```

### Major Subsystems

| Subsystem | Location | Language | Responsibility |
|-----------|----------|----------|----------------|
| Mobile App | `src/MobileI2VConsole/` | C# (.NET MAUI) | UI, DI, ONNX inference, video encoding |
| Model Conversion | `scripts/convert/` | Python 3 | PyTorch → ONNX export, manifest generation |
| Unit Tests | `tests/MobileI2VConsole.Tests/` | C# (xUnit) | Service + model unit tests |
| Python Tests | `scripts/convert/tests/` | Python (pytest) | Regression test for Turbo-VAED |
| Notebook | `scripts/notebooks/` | Jupyter | Colab-based ONNX converter |

---

## Data Flow

### 1. User Input → Generation Request

```
User picks image (gallery/camera) ──┐
                                    ├──→ HomeViewModel.PickImageAsync()
User types or selects prompt ───────┘        │
                                             │
                                    GenerationRequest {
                                        ImagePath: string,
                                        Prompt?: string,
                                        Width: 1280, Height: 720,
                                        FrameCount: 17, Fps: 17,
                                        DiffusionSteps: 2
                                    }
                                             │
                                    Shell.GoToAsync("generation")
                                             │
                                             ▼
                                    GenerationViewModel (deserializes request)
```

### 2. Inference Pipeline (6 steps)

The `InferenceOrchestrator` (`src/MobileI2VConsole/Services/InferenceOrchestrator.cs`) runs 6 sequential steps with progress reporting:

| Step | Model | Input → Output | Shape Transformation | Progress Range |
|------|-------|---------------|---------------------|----------------|
| 0 | — | Load all 4 models | — | 0% |
| 1 | VAE Encoder | RGBA image → latent | `[1,3,512,512]` → `[1,4,64,64]` | 10% |
| 2 | Qwen2-0.5B | text → embeddings | `[1,seq_len]` → `[1,seq_len,1024]` | 15% |
| 3 | MobileI2V UNet (step 1) | latent+emb+noise → denoised latent | `[1,4,64,64]` → `[1,4,64,64]` | 15–50% |
| 4 | MobileI2V UNet (step 2) | latent+emb → 17 latent frames | `[1,4,64,64]` → `[17,4,64,64]` | 50–85% |
| 5 | Turbo-VAED Decoder | 17 latents → 17 RGBA frames | `[17,4,64,64]` → 17×`[3,720,1280]` | 85–95% |
| 6 | — | float32 RGB → byte RGBA → NV12 → H.264 MP4 | — | 95–100% |

**Note:** As of the current codebase, the ONNX inference calls in `InferenceOrchestrator.cs` contain placeholder implementations (return dummy data) — lines 189–203, 225–236, 268–282, and 302–326 all have `TODO` markers. The actual ONNX sessions are created in `ModelManagerService.LoadModelAsync()` (line 157), but `InferenceOrchestrator` does not yet retrieve sessions from `IModelManager` — see the `[Obsolete]` method at line 381.

### 3. Output → Result

```
rawFrames (byte[][]) ──→ AndroidVideoEncoder.EncodeFramesAsync()
                              │
                         ┌─────▼──────┐
                         │ ConvertRgbaToNv12() │
                         └─────┬──────┘
                               │
                         ┌─────▼──────┐
                         │ MediaCodec  │  H.264 @ 4 Mbps, 17 FPS
                         │  + Muxer    │  → video_YYYYMMDDHHmmss.mp4
                         └─────┬──────┘
                               │
                         ResultViewModel.VideoPath
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
               Preview     Share      Save to Gallery
```

### 4. Model Conversion Pipeline

```
convert_all.py --output ./models/ [--optimize]
│
├── Phase 1: VAE Encoder
│   Loads AutoencoderKLLTXVideo from "Lightricks/LTX-Video"
│   Wraps with VAEEncoderWrapper (handles 4D→5D→4D)
│   Exports vae_encoder.onnx [1,3,H,W] → [1,128,H/8,W/8]
│
├── Phase 2: Qwen2-0.5B Text Encoder
│   Loads AutoModel from "Qwen/Qwen2-0.5B"
│   Strips LM head via Qwen2EncoderWrapper
│   Exports qwen2_encoder.onnx [1,seq_len] → [1,seq_len,1024]
│
├── Phase 3: MobileI2V UNet
│   Downloads hybrid_371.pth from "hustvl/MobileI2V"
│   Creates Mobiledit-300M model (vendored mobiledit.py)
│   Monkey-patches Attention for memory-efficient SDPA
│   Exports mobilei2v_unet.onnx
│
├── Phase 4: Turbo-VAED Decoder
│   Downloads config + checkpoint from "hustvl/Turbo-VAED"
│   Builds decoder via build_turbo_vaed_decoder()
│   Exports turbo_vaed.onnx [B,C,T,H,W] → [B,3,T*8,H*32,W*32]
│
└── Post-processing
    ├── Optional: compute_ort_export() (mobile-optimized .ort)
    └── build_manifest() → models_manifest.json (SHA256 + sizes)
```

---

## Key Modules

### `src/MobileI2VConsole/` — .NET MAUI Android App

#### `Models/` — Data Contracts
All located at `src/MobileI2VConsole/Models/`.

| File | Type | Key Fields | Purpose |
|------|------|-----------|---------|
| `GenerationRequest.cs` | `record` | `ImagePath`, `Prompt?`, `Width=1280`, `Height=720`, `FrameCount=17`, `Fps=17`, `DiffusionSteps=2` | Input parameters for a generation job |
| `GenerationResult.cs` | `record` | `VideoPath`, `Elapsed`, `Success=true`, `ErrorMessage?` | Output of a completed generation |
| `GenerationProgress.cs` | `record` | `StepName`, `Progress`, `OverallProgress`, `CurrentFrame`, `TotalFrames` | Pipeline progress for UI updates |
| `ModelStatus.cs` | `record` | `ModelName`, `DownloadProgress`, `IsDownloaded`, `IsLoaded`, `ErrorMessage?` | Per-model download and load status |
| `PromptTemplate.cs` | `record` | `Name`, `Prompt`, `Icon` | Predefined motion description prompts |

#### `Services/` — Business Logic

| Interface | Implementation | Responsibility | Registration |
|-----------|---------------|----------------|-------------|
| `IFileService` | `FileService` | App directory paths (`Personal/models/`, `output/`, `temp/`), directory init, temp cleanup | Singleton (line 26, `MauiProgram.cs`) |
| `IModelManager` | `ModelManagerService` | Download models from HuggingFace (with SHA256 verification + retry), manage ONNX `InferenceSession` lifecycle, NNAPI acceleration | Singleton (line 27) |
| `IMediaPickerService` | `MediaPickerService` | Wrap MAUI `MediaPicker` for gallery + camera access | Singleton (line 28) |
| `IInferenceOrchestrator` | `InferenceOrchestrator` | Full pipeline: load models → VAE encode → Qwen2 encode → UNet denoise → Turbo-VAED decode → post-process | Transient (line 29) |
| `IVideoEncoder` | `AndroidVideoEncoder` (Android) / `StubVideoEncoder` (other) | Encode RGBA frames → H.264 MP4 via Android MediaCodec | Singleton (lines 31–35) |

**Key implementation details:**

- `ModelManagerService.cs` (lines 22–28) registers 4 models with HuggingFace URLs, downloads with retry-up-to-3 and SHA256 verification
- `InferenceOrchestrator.cs` defines precise progress boundaries (lines 25–34): loading=0%, encoding image=10%, encoding text=15%, diffusion=15–85%, decoding=85–95%, saving=95–100%
- `AndroidVideoEncoder.cs` uses `MediaCodec` H.264 encoder with 4 Mbps bitrate, I-frame interval of 1s, and custom `ConvertRgbaToNv12()` for YUV420 conversion (lines 171–210)

#### `ViewModels/` — MVVM ViewModels

| File | Base Class | Key Bindable Properties | Key Commands |
|------|-----------|----------------------|--------------|
| `HomeViewModel.cs` (line 10) | `ObservableObject` | `SelectedImagePath`, `PromptText`, `ImagePreview`, `IsModelDownloaded`, `DownloadProgress` | `PickImage`, `CaptureImage`, `ApplyTemplate`, `DownloadModels`, `Generate`, `CheckModelStatus` |
| `GenerationViewModel.cs` (line 10) | `ObservableObject` | `Request` (query property), `StepName`, `Progress`, `IsRunning`, `IsCompleted`, `ResultPath`, `ErrorMessage` | `Cancel`, `ViewResult` |
| `ResultViewModel.cs` (line 8) | `ObservableObject` | `VideoPath` (query property), `VideoSource` | `ShareVideo`, `SaveToGallery`, `GenerateNew` |

All ViewModels use CommunityToolkit.Mvvm source generators (`[ObservableProperty]`, `[RelayCommand]`).

#### `Views/` — XAML Pages

| File | ViewModel | Purpose |
|------|-----------|---------|
| `HomePage.xaml` `.cs` | `HomeViewModel` | Image picker, prompt input, template chips, model download, generate button |
| `GenerationPage.xaml` `.cs` | `GenerationViewModel` | Step name, progress bar, frame counter, cancel button |
| `ResultPage.xaml` `.cs` | `ResultViewModel` | Video playback (MediaElement), share, save, generate new |

Shell navigation routes: HomePage (default route `/`), `"generation"` → GenerationPage, `"result"` → ResultPage (`AppShell.xaml.cs`, lines 10–11).

#### `Converters/` — XAML Value Converters

| File | Logic | Purpose |
|------|-------|---------|
| `InverseBoolConverter.cs` (line 12) | `!bool` | Show when not loading |
| `NotNullToBoolConverter.cs` (line 13) | `value != null` | Show when property is set |

#### `Platforms/Android/` — Android-Specific

| File | Purpose |
|------|---------|
| `MainActivity.cs` | Android activity entry point, `MauiAppCompatActivity` |
| `MainApplication.cs` | Application class, calls `MauiProgram.CreateMauiApp()` |
| `AndroidVideoEncoder.cs` | Android MediaCodec H.264 encoder + MP4 muxer |

### `scripts/convert/` — Python ONNX Conversion Pipeline

#### `convert_all.py` — Orchestrator
- Entry point: `main()` (line 85)
- Accepts `--output`, `--optimize`, `--device`, `--verbose`
- Runs 4 phases sequentially, each calling individual converter functions
- Post-processing: ORT export (optional), SHA256 manifest generation (`build_manifest()`, line 55)
- Outputs: `vae_encoder.onnx`, `qwen2_encoder.onnx`, `mobilei2v_unet.onnx`, `turbo_vaed.onnx`, `models_manifest.json`

#### Per-Converter Modules

| File | Model Source | ONNX Input Shapes | Technical Notes |
|------|-------------|-------------------|-----------------|
| `convert_vae_encoder.py` | `Lightricks/LTX-Video` (HF) | `[1,3,720,1280]` float32 | Wraps 3D VAE (5D) for 4D input; outputs `[1,128,64,80]` |
| `convert_qwen2_encoder.py` | `Qwen/Qwen2-0.5B` (HF) | `input_ids [1,77]` int64 + `attention_mask [1,77]` int64 | Strips LM head; SDPA attention; outputs `[1,77,1024]` |
| `convert_mobilei2v_unet.py` | `hustvl/MobileI2V` (HF) hybrid_371.pth | `latent [1,128,17,32,32]`, `text_emb [1,300,896]`, `timestep [1]` | 270M params; Monkey-patches `Attention.forward` for memory-efficient SDPA; `pos_embed` mismatch handling |
| `convert_turbo_vaed.py` | `hustvl/Turbo-VAED` (HF) + GitHub config | `latent [1,128,5,23,40]` | 3 variants (LTX, CogVideo5B, HunyuanVideo); strips "decoder." prefix from state dict |

#### `common/` — Shared Utilities

| File | Key Functions | Purpose |
|------|--------------|---------|
| `model_utils.py` | `get_device()`, `get_torch_dtype()`, `export_onnx()`, `verify_onnx()`, `download_repo()`, `compute_ort_export()` | Device detection (CUDA/CPU), ONNX export with `dynamo=False` for PyTorch 2.12+ compatibility, ONNX Runtime verification, selective HF downloads |
| `ort_config.py` | `optimize_onnx_model()`, `convert_to_fp16()`, `create_session_options()` | ONNX graph optimization, FP16 weight conversion, mobile session config |

Key detail in `model_utils.py`: `_patch_neg_transpose()` (line 193) fixes invalid `-1` perm values in Transpose nodes emitted by the legacy ONNX exporter.

#### `models/` — Vendored Model Architectures

| File | Classes | Lines | Description |
|------|---------|-------|-------------|
| `mobiledit.py` | `MobileditONNXWrapper`, `Attention`, `SanaBlock`, `Mobiledit`, `mobiledit_300m_P1_D16()` | 1431 | Complete model architecture with vendored dependencies (no xformers/triton) |
| `turbo_vaed_model.py` | `TurboVAEDDecoder3d`, `TurboVAEDConv2dSplitUpsampler`, `RMSNorm`, `build_turbo_vaed_decoder()` | 789 | Self-contained Turbo-VAED with custom conv blocks |
| `__init__.py` | Exports `Mobiledit`, `mobiledit_300m_P1_D16`, `build_turbo_vaed_decoder`, `TurboVAEDDecoder3d` | 2 | Package exports |

### `tests/` — Test Projects

#### C# Tests (`tests/MobileI2VConsole.Tests/`)

| File | Test Framework | Coverage | Key Test Classes |
|------|---------------|----------|-----------------|
| `MobileI2VConsole.Tests.csproj` | xUnit 2.9.3, Moq 4.20.72, coverlet 6.0.4 | — | — |
| `Models/ModelRecordTests.cs` | xUnit `[Fact]` | `GenerationRequest` defaults, `GenerationResult` defaults, `GenerationProgress` tracking, `PromptTemplate` storage |
| `Services/FileServiceTests.cs` | xUnit `[Fact]` | Directory creation, file size (nonexistent), cleanup doesn't throw |
| `Services/ModelManagerServiceTests.cs` | xUnit `[Fact]` + Moq `Mock<IFileService>` | 4 model statuses initialized, download/load false initially, path format, load returns false for missing file |

#### Python Tests (`scripts/convert/tests/`)

| File | Framework | Test |
|------|-----------|------|
| `test_turbo_vaed_bug.py` | pytest | Regression: `TurboVAEDDecoder3d` instantiates successfully with default `inject_noise` (fixed from 4→5 elements) |

### `scripts/notebooks/`

| File | Purpose |
|------|---------|
| `MobileI2V_ONNX_Converter.ipynb` | Colab notebook for ONNX model conversion (alternative to CLI) |

---

## Design Patterns

| Pattern | Usage | Location |
|---------|-------|----------|
| **MVVM** | UI architecture with data binding, source-generated observable properties and relay commands | `ViewModels/`, `Views/` |
| **Dependency Injection** | All services and ViewModels registered in `MauiProgram.CreateMauiApp()` | `MauiProgram.cs` (lines 26–49) |
| **Interface/Implementation Separation** | Every service has an `I*` interface and concrete `*Service` class | `Services/` |
| **Pipeline** | 6-step sequential inference pipeline with progress boundaries | `InferenceOrchestrator.cs` (lines 25–34, 52–124) |
| **Strategy** | Platform-specific video encoder swapped via `#if ANDROID` | `MauiProgram.cs` (lines 31–35) |
| **Wrapper/Adapter** | Python wrapper classes adapt 3D/5D models for 2D/4D ONNX export | `VAEEncoderWrapper` (convert_vae_encoder.py:45), `Qwen2EncoderWrapper` (convert_qwen2_encoder.py:45), `MobileditONNXWrapper` (convert_mobilei2v_unet.py:158), `TurboVAEDWrapper` (convert_turbo_vaed.py:85) |
| **Facade** | `convert_all.py` orchestrates 4 sub-converters + post-processing | `convert_all.py` `main()` (line 85) |
| **Registry** | 4 model definitions registered with HuggingFace URLs and order | `ModelManagerService.cs` (lines 23–28) |
| **Progress Reporting** | `IProgress<T>` used throughout for pipeline step progress | `InferenceOrchestrator.cs` method `ReportProgress()` (line 392) |

## Dependencies Between Modules

```
App.xaml.cs
    └── AppShell.xaml.cs
            ├── Views/HomePage.xaml.cs ──► ViewModels/HomeViewModel.cs
            │                                   ├── Services/IFileService
            │                                   ├── Services/IModelManager
            │                                   └── Services/IMediaPickerService
            ├── Views/GenerationPage.xaml.cs ──► ViewModels/GenerationViewModel.cs
            │                                       ├── Services/IInferenceOrchestrator
            │                                       ├── Services/IVideoEncoder
            │                                       └── Services/IFileService
            └── Views/ResultPage.xaml.cs ────► ViewModels/ResultViewModel.cs

Services/IInferenceOrchestrator ──► InferenceOrchestrator
    ├── Services/IModelManager (load/unload ONNX sessions)
    └── Services/IFileService (paths, temp files)

Services/IModelManager ──► ModelManagerService
    ├── Services/IFileService (model directory)
    ├── Microsoft.ML.OnnxRuntime (InferenceSession)
    └── Models/ (ModelStatus records)

Services/IVideoEncoder ──► AndroidVideoEncoder
    └── Android.Media.MediaCodec + MediaMuxer

Models/ ──► (used by Services, ViewModels, Views)
    ├── GenerationRequest
    ├── GenerationResult
    ├── GenerationProgress
    ├── ModelStatus
    └── PromptTemplate

Python conversion (scripts/convert/):
convert_all.py
    ├── convert_vae_encoder.py ──► common/model_utils.py
    ├── convert_qwen2_encoder.py ──► common/model_utils.py
    ├── convert_mobilei2v_unet.py ──► common/model_utils.py, models/mobiledit.py
    ├── convert_turbo_vaed.py ──► common/model_utils.py, models/turbo_vaed_model.py
    └── common/ort_config.py
```

## Entry Points

### Primary: .NET MAUI App

| Entry Point | File | Arguments | What It Does |
|-------------|------|-----------|-------------|
| App builder | `MauiProgram.cs` `CreateMauiApp()` (line 12) | None (called by framework) | Registers all DI services, ViewModels, pages, converters; configures fonts, CommunityToolkit |
| Application | `App.xaml.cs` | None | Initializes app shell (`new AppShell()` → line 12) |
| Shell | `AppShell.xaml.cs` | None | Registers Shell navigation routes: `"generation"`, `"result"` |
| Android entry | `Platforms/Android/MainApplication.cs` `CreateMauiApp()` (line 14) | None (called by Android runtime) | Returns `MauiProgram.CreateMauiApp()` |
| Android activity | `Platforms/Android/MainActivity.cs` | Android lifecycle | Standard MAUI activity with config change handling |
| HomePage | `Views/HomePage.xaml.cs` (line 7) | `HomeViewModel` (DI) | Main page — image picker, prompt input, generate |
| GenerationPage | `Views/GenerationPage.xaml.cs` (line 7) | `GenerationViewModel` (DI) | Progress display during generation |
| ResultPage | `Views/ResultPage.xaml.cs` (line 7) | `ResultViewModel` (DI) | Video preview, share, save |

### Secondary: Python Model Conversion CLI

| Script | Arguments | What It Does |
|--------|-----------|-------------|
| `python convert_all.py` | `--output`, `--optimize`, `--device`, `--verbose` | Converts all 4 models sequentially, optionally exports ORT format, writes manifest |
| `python convert_vae_encoder.py` | `--output`, `--height`, `--width`, `--verbose`, `--device` | Exports VAE encoder to ONNX |
| `python convert_qwen2_encoder.py` | `--output`, `--max-seq-len`, `--verbose`, `--device` | Exports Qwen2-0.5B text encoder to ONNX |
| `python convert_mobilei2v_unet.py` | `--output`, `--verbose`, `--device` | Downloads checkpoint, loads Mobiledit model, exports UNet to ONNX |
| `python convert_turbo_vaed.py` | `--output`, `--variant`, `--height`, `--width`, `--num-frames`, `--verbose`, `--device` | Downloads config + checkpoint, exports Turbo-VAED decoder to ONNX |

### NuGet Dependencies (from `MobileI2VConsole.csproj`)

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.Maui.Controls` | 10.0.80 | MAUI cross-platform UI framework |
| `Microsoft.ML.OnnxRuntime` | 1.27.0 | ONNX model inference engine |
| `CommunityToolkit.Mvvm` | 8.4.2 | MVVM source generators (`[ObservableProperty]`, `[RelayCommand]`) |
| `CommunityToolkit.Maui` | 14.2.0 | MAUI community controls |
| `CommunityToolkit.Maui.MediaElement` | 10.0.0 | Video playback control |
| `SkiaSharp` | 4.148.0 | Image loading and processing |

### Python Dependencies (from `requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1.0 (≥2.7.0 for CUDA 12.6) | PyTorch model framework |
| `onnx` | ≥1.15.0 | ONNX format handling |
| `onnxruntime` | ≥1.18.0 | ONNX Runtime verification (CPU; use `onnxruntime-gpu≥1.19.0` for GPU) |
| `transformers` | ≥4.44.0 | Qwen2 model loading |
| `diffusers` | ≥0.30.0 | VAE (AutoencoderKLLTXVideo) loading |
| `huggingface-hub` | ≥0.20.0 | HuggingFace model download |
