# Phase 1 — MAUI Android App Architecture

**Status:** Proposed  
**Date:** 2026-07-01  
**Applies to:** Phase 1 (scaffold, model management, inference pipeline, video encoding, UI)

---

## 1. Solution Structure

`
MobileI2VConsole/
├── MobileI2VConsole.sln                      # .NET 8 solution
├── src/
│   └── MobileI2VConsole/                     # MAUI Android project
│       ├── MobileI2VConsole.csproj
│       ├── App.xaml / App.xaml.cs             # Application entry, service registration
│       ├── AppShell.xaml / AppShell.xaml.cs   # Shell navigation (3 routes)
│       ├── MauiProgram.cs                     # DI container setup
│       ├── Constants.cs                       # App-wide constants (paths, model IDs)
│       │
│       ├── Models/                            # Domain models
│       │   ├── GenerationRequest.cs
│       │   ├── GenerationResult.cs
│       │   ├── GenerationProgress.cs
│       │   ├── PromptTemplate.cs
│       │   └── ModelInfo.cs
│       │
│       ├── Services/                          # Business logic & platform services
│       │   ├── Interfaces/
│       │   │   ├── IModelManagerService.cs
│       │   │   ├── IInferenceOrchestrator.cs
│       │   │   ├── IVideoEncoderService.cs
│       │   │   ├── IImagePickerService.cs
│       │   │   └── IFileService.cs
│       │   ├── ModelManagerService.cs
│       │   ├── InferenceOrchestrator.cs
│       │   ├── VideoEncoderService.cs         # Android-specific (MediaCodec)
│       │   ├── ImagePickerService.cs
│       │   └── FileService.cs
│       │
│       ├── ViewModels/                        # MVVM view models
│       │   ├── HomeViewModel.cs
│       │   ├── GenerationViewModel.cs
│       │   └── ResultViewModel.cs
│       │
│       ├── Views/                             # XAML pages
│       │   ├── HomePage.xaml / .cs
│       │   ├── GenerationPage.xaml / .cs
│       │   └── ResultPage.xaml / .cs
│       │
│       ├── Controls/                          # Reusable custom controls
│       │   ├── PromptChipBar.xaml / .cs
│       │   └── ProgressIndicator.xaml / .cs
│       │
│       ├── Converters/                        # XAML value converters
│       │   ├── BoolToVisibilityConverter.cs
│       │   └── ProgressToPercentConverter.cs
│       │
│       ├── Resources/
│       │   ├── Fonts/
│       │   ├── Images/
│       │   │   ├── icon_home.svg
│       │   │   ├── icon_gallery.svg
│       │   │   └── icon_camera.svg
│       │   ├── Raw/                            # Bundled assets
│       │   │   └── prompt_templates.json
│       │   └── Styles/
│       │       ├── Colors.xaml
│       │       └── Styles.xaml
│       │
│       └── Platforms/
│           └── Android/
│               ├── AndroidManifest.xml
│               ├── MainActivity.cs
│               ├── MainApplication.cs
│               └── Services/                   # Android-specific service impl.
│                   └── AndroidVideoEncoder.cs  # Native MediaCodec wrapper
│
├── scripts/
│   └── convert/                                # Python ONNX conversion scripts
│       ├── requirements.txt
│       ├── common/
│       │   ├── __init__.py
│       │   ├── model_utils.py                  # Shared ONNX export utilities
│       │   └── ort_config.py                   # ORT quantization & optimization
│       ├── convert_qwen2_encoder.py            # Qwen2-0.5B → ONNX
│       ├── convert_mobilei2v_unet.py           # MobileI2V UNet → ONNX
│       ├── convert_turbo_vaed.py               # Turbo-VAED → ONNX
│       ├── convert_vae_encoder.py              # VAE encoder (for input image)
│       └── validate_models.py                  # Validate exported ONNX models
│
├── tests/
│   ├── MobileI2VConsole.Tests/                # Unit tests (.NET)
│   │   ├── MobileI2VConsole.Tests.csproj
│   │   ├── Services/
│   │   │   ├── ModelManagerServiceTests.cs
│   │   │   └── InferenceOrchestratorTests.cs
│   │   └── ViewModels/
│   │       ├── HomeViewModelTests.cs
│   │       └── GenerationViewModelTests.cs
│   │
│   └── MobileI2VConsole.IntegrationTests/     # Integration tests
│       └── ...
│
└── .config/
    └── dotnet-tools.json                       # dotnet CLI tool manifest
`

---

## 2. Namespace / Class Hierarchy

### 2.1 Project Namespace: MobileI2VConsole

| Namespace | Contents | Responsibility |
|-----------|----------|---------------|
| MobileI2VConsole.Models | GenerationRequest, GenerationResult, GenerationProgress, PromptTemplate, ModelInfo | Domain data contracts |
| MobileI2VConsole.Services | Interface definitions + service implementations | Business logic orchestration |
| MobileI2VConsole.Services.Interfaces | IModelManagerService, IInferenceOrchestrator, IVideoEncoderService, IImagePickerService, IFileService | Service contracts for DI |
| MobileI2VConsole.ViewModels | HomeViewModel, GenerationViewModel, ResultViewModel | MVVM view models (CommunityToolkit.Mvvm) |
| MobileI2VConsole.Views | HomePage, GenerationPage, ResultPage | XAML UI pages |
| MobileI2VConsole.Controls | PromptChipBar, ProgressIndicator | Reusable custom controls |
| MobileI2VConsole.Converters | BoolToVisibilityConverter, ProgressToPercentConverter | XAML binding converters |
| MobileI2VConsole.Platforms.Android.Services | AndroidVideoEncoder | Android-native MediaCodec wrapper |

### 2.2 Model Classes

`csharp
// Models/GenerationRequest.cs
public record GenerationRequest
{
    public string ImagePath { get; init; }
    public string Prompt { get; init; }
    public int NumSteps { get; init; } = 2;
    public int NumFrames { get; init; } = 17;
    public int Width { get; init; } = 1280;
    public int Height { get; init; } = 720;
}

// Models/GenerationResult.cs
public record GenerationResult
{
    public string VideoPath { get; init; }
    public int FrameCount { get; init; }
    public TimeSpan Duration { get; init; }
    public long FileSizeBytes { get; init; }
    public bool Success { get; init; }
    public string? ErrorMessage { get; init; }
}

// Models/GenerationProgress.cs
public record GenerationProgress
{
    public string Stage { get; init; }
    public double PercentComplete { get; init; }
    public string? Detail { get; init; }
}

// Models/PromptTemplate.cs
public record PromptTemplate
{
    public string Name { get; init; }
    public string Prompt { get; init; }
    public string Icon { get; init; }
}

// Models/ModelInfo.cs
public record ModelInfo
{
    public string ModelId { get; init; }
    public string HuggingFaceUrl { get; init; }
    public string LocalPath { get; init; }
    public long ExpectedSizeBytes { get; init; }
    public bool IsDownloaded { get; init; }
    public bool IsLoaded { get; init; }
}
`

### 2.3 Service Interfaces

`csharp
public interface IModelManagerService
{
    Task EnsureModelsDownloadedAsync(IProgress<GenerationProgress>? progress = null, CancellationToken ct = default);
    Task LoadModelsAsync(IProgress<GenerationProgress>? progress = null, CancellationToken ct = default);
    void UnloadModels();
    Task<bool> AreModelsDownloadedAsync();
    IReadOnlyList<ModelInfo> GetModelInfos();
    OrtSession GetSession(string modelKey);
}

public interface IInferenceOrchestrator
{
    Task<GenerationResult> GenerateAsync(GenerationRequest request, IProgress<GenerationProgress>? progress = null, CancellationToken ct = default);
    bool CanGenerate();
}

public interface IVideoEncoderService
{
    Task<string> EncodeFramesAsync(IReadOnlyList<byte[]> frames, int width, int height, int fps, string outputPath, CancellationToken ct = default);
}

public interface IImagePickerService
{
    Task<FileResult?> PickImageAsync(bool useCamera = false);
}

public interface IFileService
{
    string GetModelsDirectory();
    string GetOutputDirectory();
    string GetCacheDirectory();
    long GetFreeDiskSpaceBytes();
}
`

### 2.4 ViewModel Classes

`csharp
public partial class HomeViewModel : ObservableObject
{
    [ObservableProperty] private string? selectedImagePath;
    [ObservableProperty] private string customPrompt = "";
    [ObservableProperty] private PromptTemplate? selectedTemplate;
    [ObservableProperty] private bool isModelReady;
    [ObservableProperty] private bool isBusy;
    [ObservableProperty] private string modelStatusText = "Checking models...";

    public ObservableCollection<PromptTemplate> Templates { get; }
    public ICommand PickImageCommand { get; }
    public ICommand CaptureImageCommand { get; }
    public ICommand SelectTemplateCommand { get; }
    public ICommand GenerateCommand { get; }
}

public partial class GenerationViewModel : ObservableObject
{
    [ObservableProperty] private GenerationRequest request = default!;
    [ObservableProperty] private string currentStage = "Starting...";
    [ObservableProperty] private double progressValue;
    [ObservableProperty] private bool isCompleted;
    [ObservableProperty] private bool hasError;
    [ObservableProperty] private string? errorMessage;
}

public partial class ResultViewModel : ObservableObject
{
    [ObservableProperty] private GenerationResult result = default!;
    [ObservableProperty] private string? videoPath;

    public ICommand ShareCommand { get; }
    public ICommand SaveToGalleryCommand { get; }
    public ICommand GenerateNewCommand { get; }
}
`

---

## 3. Data Flow

### 3.1 End-to-End Pipeline

`
HomePage: User picks image + prompt → Generate tap
    ↓ Navigation with GenerationRequest
GenerationPage: InferenceOrchestrator.GenerateAsync()
    ├── VAE Encode: image (1280x720) → latent (4x90x160)
    ├── Qwen2 Encode: text prompt → embeddings (77x1024)
    ├── UNet 2-step denoising: latent + embeddings → denoised latents (17 frames)
    ├── Turbo-VAED Decode: latents → 17 RGBA frames (720p)
    └── MediaCodec H.264 Encode: frames → .mp4 file
    ↓ Navigation with GenerationResult
ResultPage: Preview video → Share / Save / Generate New
`

### 3.2 Progress Mapping

| Stage | % Range | Detail |
|-------|---------|--------|
| loading_models | 0–5% | Ensuring models are loaded |
| ncoding_image | 5–10% | VAE encoding source image |
| ncoding_text | 10–15% | Qwen2 encoding text prompt |
| diffusion | 15–60% | UNet denoising step 1/2 |
| decoding_frames | 60–90% | Turbo-VAED frame 1/17 → 17/17 |
| ncoding_video | 90–100% | MediaCodec H.264 encoding |

---

## 4. Model Management Service

### 4.1 Storage Layout

`
{Android.DataDir}/files/models/
├── vae-encoder/
│   └── model.onnx              # ~50MB
├── qwen2-0.5b-encoder/
│   └── model.onnx              # ~900MB
├── mobilei2v-unet/
│   └── model.onnx              # ~540MB (270M params @ fp16)
└── turbo-vaed-decoder/
    └── model.onnx              # ~50MB
`

### 4.2 Download Flow

1. **First launch**: Check free space (need 2.5 GB+), download models sequentially from HuggingFace via HttpClient with HTTP Range resume support, verify SHA256, persist completion flag
2. **Subsequent launches**: Quick integrity check (file existence + expected size), re-download only corrupted models
3. **Model manifest**: Embedded JSON with URL, SHA256, and expected size per model

### 4.3 Loading Strategy

| Model | RAM | Lifetime |
|-------|-----|----------|
| VAE Encoder | ~60 MB | Load → encode → dispose |
| Qwen2 Encoder | ~950 MB | Load → encode → dispose |
| MobileI2V UNet | ~580 MB | Load → 2 inferences → dispose |
| Turbo-VAED | ~60 MB | Load → 17 inferences → dispose |
| **Peak concurrent** | **~1.65 GB** | All loaded during decode handoff |

Session options: ORT_SEQUENTIAL execution, ORT_ENABLE_ALL graph optimization, prefer DML (DirectML) provider, fallback to CPU.

---

## 5. Inference Orchestrator

### 5.1 Core Pipeline

`
GenerateAsync(request, progress, ct):
  1. LoadAndPreprocessImage(imagePath) → float[3,720,1280] tensor
  2. RunInference("vae-encoder", imageTensor) → float[4,90,160] latent
  3. EncodeTextPrompt(prompt) → float[77,1024] embeddings
  4. RunDiffusionAsync(latent, embeddings, 2 steps) → float[17,4,90,160] frames
  5. DecodeFramesAsync(frameLatents) → List<byte[1280*720*4]> RGBA frames
  6. videoEncoder.EncodeFramesAsync(frames, 1280, 720, 17, outputPath) → .mp4
`

### 5.2 2-Step Diffusion

- Step 1 (t=1.0): Predict noise residual, apply denoising
- Step 2 (t=0.0): Predict clean latent directly (turbo shortcut)
- Output expanded from single latent [1,4,90,160] → 17-frame latent [17,4,90,160] by applying small temporal shifts per frame

### 5.3 Turbo-VAED Decode

- Decodes each of 17 frames independently
- Converts float32 RGB [-1, 1] → byte RGBA [0, 255]
- Returns list of 17 RGBA byte arrays (1280×720×4 bytes each)

---

## 6. Video Encoder Service (MediaCodec)

### 6.1 Android Implementation

Platform-specific AndroidVideoEncoder wrapping Android.Media.MediaCodec:
- H.264 AVC codec, 720p resolution, 17 FPS
- 4 Mbps bitrate, 1s keyframe interval
- MediaMuxer to produce MP4 container
- Frames fed as input buffers via DequeueInputBuffer / QueueInputBuffer
- Drains output via DequeueOutputBuffer → writes to MediaMuxer

### 6.2 DI Registration

`csharp
#if ANDROID
builder.Services.AddSingleton<IVideoEncoderService, AndroidVideoEncoder>();
#endif
`
All services and ViewModels registered as singletons or transients in MauiProgram.cs using CommunityToolkit.Maui and Microsoft.ML.OnnxRuntime.

---

## 7. UI Architecture

### 7.1 Shell Navigation

- **HomePage** (route: home): Default ShellContent, no tab bar
- **GenerationPage** (route: generation): Registered via Routing.RegisterRoute
- **ResultPage** (route: esult): Registered via Routing.RegisterRoute

Navigation flow: Home → generation → result → //home

### 7.2 Query Parameters

ViewModels use [QueryProperty] attributes to receive serialized GenerationRequest and GenerationResult as JSON strings via Shell.GoToAsync dictionary parameters.

### 7.3 Screen Layouts

**HomePage**: Image preview area, [Pick from Gallery] + [Camera] buttons, text Entry for prompt, horizontal PromptChipBar (Gentle Motion, Zoom In, Flowing, Dramatic, Cinematic Pan), [Generate] CTA button (disabled until image selected), model status footer.

**GenerationPage**: Live frame preview (optional), ProgressBar bound to ProgressValue, stage text label, percentage label, [Cancel] button.

**ResultPage**: MediaElement (auto-loop video preview), video metadata labels (duration, resolution, size), [Share] / [Save to Gallery] / [Generate New] buttons.

---

## 8. Conversion Pipeline (Python → ONNX)

### 8.1 Scripts

| Script | Model | Input Shapes | Output Shapes |
|--------|-------|-------------|---------------|
| convert_vae_encoder.py | VAE Encoder | [1,3,720,1280] f32 | [1,4,90,160] f32 |
| convert_qwen2_encoder.py | Qwen2-0.5B | [1,77] int64 | [1,77,1024] f32 |
| convert_mobilei2v_unet.py | MobileI2V UNet | [1,4,90,160] + [1,77,1024] + [1] f32 | [1,4,90,160] f32 |
| convert_turbo_vaed.py | Turbo-VAED | [1,4,90,160] f32 | [1,3,720,1280] f32 |
| alidate_models.py | All | Random inputs | Shape verification |

### 8.2 Shared Utilities

common/model_utils.py provides xport_onnx() helper (model.eval(), torch.onnx.export, checker.check_model).
common/ort_config.py provides fp16 conversion and graph optimization config.

### 8.3 Dependencies

`
torch>=2.1.0, onnx>=1.15.0, onnxruntime>=1.19.0,
transformers>=4.40.0, diffusers>=0.29.0, numpy, onnxruntime-tools
`

---

## 9. Key Dependencies

### 9.1 NuGet Packages

| Package | Version | Purpose |
|---------|---------|---------|
| Microsoft.ML.OnnxRuntime | ≥1.19.0 | ONNX managed bindings (Android .so included) |
| CommunityToolkit.Mvvm | ≥8.3.0 | Source-generated MVVM ([ObservableProperty], [RelayCommand]) |
| CommunityToolkit.Maui | ≥9.0.0 | MediaElement, Toast, MAUI helpers |

### 9.2 Android APIs

ndroid.media.MediaCodec (H.264 encode), ndroid.media.MediaMuxer (MP4 muxing), ndroid.graphics.Bitmap (image decode), Android.Content.ContentValues (MediaStore gallery save).

### 9.3 Permissions

INTERNET, ACCESS_NETWORK_STATE, READ_MEDIA_IMAGES (API 33+), READ_EXTERNAL_STORAGE (≤ API 32), CAMERA (optional feature).

---

## 10. Error Handling

| Scenario | Handling |
|----------|----------|
| No internet | Show error + retry; resume partial downloads via HTTP Range |
| Insufficient disk | Pre-check 2.5 GB free; user prompt |
| Model corruption | SHA256 verification; per-model re-download |
| OOM | Catch OrtException; unload unused models |
| Image invalid | Resize/crop to 720p; validation at boundary |
| User cancels | CancellationToken; clean up temp files |
| MediaCodec failure | Catch exception; return error result |
| Permission denied | Show rationale dialog; re-request |

---

## 11. Performance Budget

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Model download (first launch) | < 5 min | HttpClient timing |
| Model loading | < 10 s | Stopwatch |
| VAE encode | < 2 s | ONNX inference |
| Qwen2 encode | < 3 s | ONNX inference |
| UNet 2-step diffusion | < 15 s | ONNX inference (×2) |
| Turbo-VAED decode (17 frames) | < 10 s | ONNX inference (×17) |
| MediaCodec encode | < 3 s | MediaCodec timing |
| **Total generation** | **< 35 s** | Full pipeline |
| Peak memory | < 1.8 GB | Android Profiler |
| Output file | 1–2 MB | File size |

---

## 12. Architecture Decision Records

| ID | Decision |
|----|----------|
| D-01 | Android min API 26 |
| D-02 | Models from HuggingFace, cached to app files dir |
| D-03 | Gallery pick primary, camera secondary |
| D-04 | H.264 .mp4, 720p, 17 frames @ 17 FPS |
| D-05 | Custom text prompt + 5 template chips |
| D-06 | Three-screen Shell navigation (no tabs) |
| D-07 | Python conversion scripts in scripts/convert/ |
| D-08 | Microsoft.ML.OnnxRuntime for ONNX inference |
| D-09 | CommunityToolkit.Mvvm for MVVM |
| D-10 | Android.Media.MediaCodec for H.264 encode |
| D-11 | Float32 ONNX tensors (avoid fp16 precision issues) |
| D-12 | Sequential model load/unload per pipeline stage |
| D-13 | RGBA byte arrays across managed/native boundary |
| D-14 | IProgress<GenerationProgress> for pipeline feedback |
