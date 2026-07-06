# Tech Stack

## Languages

| Language | Usage | Estimate |
|----------|-------|----------|
| C# | Main MAUI Android application | ~65% |
| Python 3.10+ | ONNX model conversion pipeline | ~20% |
| XAML | UI layout definitions | ~10% |
| PowerShell | Utility scripts (`scripts/fd-status.ps1`) | ~3% |
| Jupyter Notebook | Model conversion exploration (`scripts/notebooks/MobileI2V_ONNX_Converter.ipynb`) | ~2% |

## Frontend (Mobile)

| Component | Technology | Version | Source |
|-----------|-----------|---------|--------|
| UI Framework | .NET MAUI | 10.0.80 | `src/MobileI2VConsole/MobileI2VConsole.csproj` line 38 |
| UI Language | XAML | — | `.xaml` files in `src/MobileI2VConsole/Views/` |
| MVVM Toolkit | CommunityToolkit.Mvvm | 8.4.2 | `src/MobileI2VConsole/MobileI2VConsole.csproj` line 40 |
| MAUI Toolkit | CommunityToolkit.Maui | 14.2.0 | `src/MobileI2VConsole/MobileI2VConsole.csproj` line 41 |
| Media Element | CommunityToolkit.Maui.MediaElement | 10.0.0 | `src/MobileI2VConsole/MobileI2VConsole.csproj` line 42 |
| Navigation | .NET MAUI Shell | — | `src/MobileI2VConsole/AppShell.xaml` |
| Fonts | OpenSans (Regular, Semibold) | — | `src/MobileI2VConsole/MauiProgram.cs` lines 21-22 |
| Graphics | SkiaSharp | 4.148.0 | `src/MobileI2VConsole/MobileI2VConsole.csproj` line 43 |

### Pages (3)

| Page | Path | Purpose |
|------|------|---------|
| HomePage | `Views/HomePage.xaml` | Image picker, prompt input, model status |
| GenerationPage | `Views/GenerationPage.xaml` | Inference pipeline progress |
| ResultPage | `Views/ResultPage.xaml` | Video playback and share |

### ViewModels (3)

| ViewModel | Path | Pattern |
|-----------|------|---------|
| HomeViewModel | `ViewModels/HomeViewModel.cs` | `ObservableObject` + `[ObservableProperty]` + `[RelayCommand]` |
| GenerationViewModel | `ViewModels/GenerationViewModel.cs` | Same pattern |
| ResultViewModel | `ViewModels/ResultViewModel.cs` | Same pattern |

## Backend (On-Device Inference)

| Component | Technology | Version | Source |
|-----------|-----------|---------|--------|
| Runtime | .NET 10.0 | `net10.0-android` | `MobileI2VConsole.csproj` line 4 |
| ML Inference | ONNX Runtime (Microsoft.ML.OnnxRuntime) | 1.27.0 | `MobileI2VConsole.csproj` line 39 |
| Image Processing | SkiaSharp | 4.148.0 | `MobileI2VConsole.csproj` line 43 |
| API Style | Interface/Service DI (MVVM) | — | `MauiProgram.cs` lines 26-49 |

There is **no cloud backend**. All inference runs fully on-device.

### ONNX Inference Pipeline (4 models)

| Model | Source | ONNX Size | Pipeline Step |
|-------|--------|-----------|---------------|
| VAE Encoder | HuggingFace `hustvl/MobileI2V` | ~50 MB | Image → 4×64×64 latent |
| Qwen2-0.5B Encoder | HuggingFace `hustvl/MobileI2V` | ~900 MB | Text prompt → 768-d embeddings |
| MobileI2V UNet | HuggingFace `hustvl/MobileI2V` | ~540 MB | 2-step turbo diffusion denoising |
| Turbo-VAED Decoder | HuggingFace `hustvl/Turbo-VAED` | ~50 MB | Latent → 720p RGBA frames (×17) |

See `src/MobileI2VConsole/Services/InferenceOrchestrator.cs` lines 58-68 for the model loading order.

### Android-Specific

| Component | Technology | Details |
|-----------|-----------|---------|
| Target Platform | Android 8.0+ (API 26+) | `MobileI2VConsole.csproj` line 17, `SupportedOSPlatformVersion` |
| Hardware Acceleration | NNAPI (optional, fallback to CPU) | `ModelManagerService.cs` line 155 |
| Video Encoding | Android MediaCodec (H.264 → MP4) | `Platforms/Android/AndroidVideoEncoder.cs` lines 12-165 |
| Output Format | H.264 MP4 @ 17 FPS, 4 Mbps bitrate | `AndroidVideoEncoder.cs` lines 39-40 |
| Model Download | HuggingFace via HTTP | `ModelManagerService.cs` lines 22-28 |

## Database & Storage

| Component | Technology | Details |
|-----------|-----------|---------|
| Primary Storage | File system (app data directory) | `IFileService` / `FileService.cs` |
| Cache | Local file cache for downloaded ONNX models | `ModelManagerService.cs` lines 53-55 |
| Manifest | `models_manifest.json` (SHA256 verification) | `ModelManagerService.cs` lines 228-243 |
| Database | **None** — everything is file-based | — |

## Infrastructure

| Component | Technology | Details |
|-----------|-----------|---------|
| Deployment Platform | Android device (APK) | `dotnet publish -f net10.0-android -c Release` |
| CI/CD | **None found** — no `.github/workflows/` or CI config files | — |
| Containerization | **None found** — no Dockerfile | — |
| Cloud Services | **None** — fully on-device, no cloud dependencies | — |

## Development Tools

| Tool | Technology | Version | Source |
|------|-----------|---------|--------|
| SDK | .NET SDK | 10.0 | `MobileI2VConsole.csproj` line 4 (`net10.0-android`) |
| IDE | Visual Studio 2022+ (Mobile development with .NET workload) | 17.8+ | `MobileI2VConsole.sln` line 4 |
| Package Manager (C#) | NuGet | — | `.csproj` PackageReference elements |
| Package Manager (Python) | pip | — | `scripts/convert/requirements.txt` |
| Test Framework | xUnit | 2.9.3 | `MobileI2VConsole.Tests.csproj` line 11 |
| Test Runner | xunit.runner.visualstudio | 3.0.2 | `MobileI2VConsole.Tests.csproj` line 12 |
| Mocking | Moq | 4.20.72 | `MobileI2VConsole.Tests.csproj` line 16 |
| Code Coverage | coverlet | 6.0.4 | `MobileI2VConsole.Tests.csproj` line 17 |
| ML Framework (for conversion) | PyTorch | ≥2.1.0 (≥2.7.0 for CUDA 12.6) | `scripts/convert/requirements.txt` line 14 |
| Python Type Checking | **None configured** — no mypy, pyright, or type checker config found | — | — |
| Linting/Formatting | **None configured** — no `.editorconfig`, `.eslintrc`, `.stylecop`, or ruff config found | — | — |

## Dependencies

### Production (C# NuGet packages)

| Package | Version | Purpose | Source File:Line |
|---------|---------|---------|-----------------|
| Microsoft.Maui.Controls | 10.0.80 | MAUI framework | `MobileI2VConsole.csproj:38` |
| Microsoft.ML.OnnxRuntime | 1.27.0 | ONNX model inference | `MobileI2VConsole.csproj:39` |
| CommunityToolkit.Mvvm | 8.4.2 | MVVM source generators | `MobileI2VConsole.csproj:40` |
| CommunityToolkit.Maui | 14.2.0 | MAUI community controls | `MobileI2VConsole.csproj:41` |
| CommunityToolkit.Maui.MediaElement | 10.0.0 | Video playback element | `MobileI2VConsole.csproj:42` |
| SkiaSharp | 4.148.0 | Image decode/resize/crop | `MobileI2VConsole.csproj:43` |

### Production (Python — model conversion)

| Package | Minimum Version | Purpose | Source File:Line |
|---------|----------------|---------|-----------------|
| torch | ≥2.1.0 (≥2.7.0 for CUDA 12.6) | PyTorch model loading and ONNX export | `scripts/convert/requirements.txt:14` |
| torchvision | ≥0.16.0 (≥0.22.0 for CUDA 12.6) | Image transforms for model export | `scripts/convert/requirements.txt:15` |
| onnx | ≥1.15.0 | ONNX model format support | `scripts/convert/requirements.txt:16` |
| onnxruntime | ≥1.18.0 (CPU) or ≥1.19.0 (GPU) | ONNX validation during conversion | `scripts/convert/requirements.txt:17` |
| transformers | ≥4.44.0 | HuggingFace model loading (Qwen2) | `scripts/convert/requirements.txt:18` |
| huggingface-hub | ≥0.20.0 | Model download from HuggingFace | `scripts/convert/requirements.txt:19` |
| diffusers | ≥0.30.0 | VAE, UNet model components | `scripts/convert/requirements.txt:20` |
| numpy | ≥1.24.0 | Numerical operations | `scripts/convert/requirements.txt:21` |
| Pillow | ≥10.0.0 | Image I/O for export testing | `scripts/convert/requirements.txt:22` |
| safetensors | ≥0.4.0 | Safe tensor serialization | `scripts/convert/requirements.txt:23` |
| opencv-python | ≥4.8.0 | Video frame processing | `scripts/convert/requirements.txt:24` |

### Development (C# NuGet — test project)

| Package | Version | Purpose | Source File:Line |
|---------|---------|---------|-----------------|
| xunit | 2.9.3 | Unit test framework | `MobileI2VConsole.Tests.csproj:11` |
| xunit.runner.visualstudio | 3.0.2 | Test runner integration | `MobileI2VConsole.Tests.csproj:12` |
| Moq | 4.20.72 | Service mocking | `MobileI2VConsole.Tests.csproj:16` |
| coverlet.collector | 6.0.4 | Code coverage collection | `MobileI2VConsole.Tests.csproj:17` |

## Key Architecture Patterns

- **MVVM** — ViewModels use CommunityToolkit.Mvvm source generators (`[ObservableProperty]`, `[RelayCommand]`)
- **Dependency Injection** — All services, ViewModels, pages registered in `MauiProgram.cs` using `Microsoft.Extensions.DependencyInjection`
- **Interface/Service separation** — Every public service has an `I{Service}` interface (`IFileService`, `IModelManager`, `IInferenceOrchestrator`, `IVideoEncoder`, `IMediaPickerService`)
- **Platform-specific implementations** — `IVideoEncoder` resolved to `AndroidVideoEncoder` on Android, `StubVideoEncoder` on other targets (via `#if ANDROID` conditional compilation)
- **Shell Navigation** — Pages registered as routes in `AppShell.xaml`, navigated via `Shell.Current.GoToAsync()`
- **Shell.FlyoutBehavior** set to `Disabled` — single-window flow with push navigation

## Test Coverage

| Test File | Tests | Source File:Line |
|-----------|-------|-----------------|
| `Services/ModelManagerServiceTests.cs` | Model manager unit tests | — |
| `Services/FileServiceTests.cs` | File service unit tests | — |
| `Models/ModelRecordTests.cs` | Model record unit tests | — |
| `ViewModels/HomeViewModelTests.cs` | HomeViewModel unit tests | — |

Test command: `dotnet test tests/MobileI2VConsole.Tests`
