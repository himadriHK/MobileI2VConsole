# Project Structure

MobileI2VConsole is a .NET MAUI Android app for on-device image-to-video generation using ONNX Runtime, paired with Python scripts for converting PyTorch models to ONNX format.

## Directory Layout

```
MobileI2VConsole/
├── .codebase/                         — Codebase documentation (auto-generated)
│   ├── AUDIT.jsonl                    — Audit log
│   ├── CODEGRAPH.md                   — Codegraph index info
│   ├── CONSTRAINTS.md                 — Architecture constraints
│   ├── DECISIONS.jsonl                — Decision log
│   ├── FAILURES.json                  — Failure replay database
│   ├── POLICIES.json                  — Self-healing policies
│   └── VERIFICATION.jsonl             — Verification log
├── .codegraph/                        — Codegraph symbol index
├── .flowdeck/                         — FlowDeck agent orchestration config
├── .opencode/                         — Opencode editor/agent configuration
├── .planning/                         — Agent planning state files
├── .vs/                               — Visual Studio settings (user-local)
├── scripts/                           — Python model conversion scripts
│   ├── convert/                       — ONNX export pipeline
│   │   ├── common/                    — Shared ONNX export utilities
│   │   │   ├── __init__.py
│   │   │   ├── model_utils.py         — Device selection, HF download, export, verify
│   │   │   └── ort_config.py          — ORT session config, FP16 conversion, optimization
│   │   ├── models/                    — Model definitions + exported ONNX files
│   │   │   ├── __init__.py
│   │   │   ├── mobiledit.py           — MobileI2V UNet architecture (vendored from hustvl/MobileI2V)
│   │   │   ├── turbo_vaed_model.py    — Turbo-VAED decoder model
│   │   │   ├── qwen2_encoder.onnx     — Exported Qwen2 text encoder
│   │   │   ├── qwen2_encoder.onnx.data
│   │   │   ├── turbo_vaed.onnx        — Exported VAE decoder
│   │   │   ├── vae_encoder.onnx       — Exported VAE encoder
│   │   │   └── mobilei2v_unet.onnx    — Exported UNet (gitignored, built from script)
│   │   ├── tests/                     — Python tests for conversion
│   │   │   ├── __init__.py
│   │   │   └── test_turbo_vaed_bug.py — Regression test for VAED shape issue
│   │   ├── model_cache/               — Downloaded HF model weights (cached)
│   │   │   ├── .cache/huggingface/    — HuggingFace cache
│   │   │   ├── .agent_harnesses.json
│   │   │   └── hybrid_371.pth         — Cached checkpoint
│   │   ├── convert_all.py             — Batch conversion of all models
│   │   ├── convert_mobilei2v_unet.py  — UNet ONNX export script
│   │   ├── convert_qwen2_encoder.py   — Qwen2 text encoder export
│   │   ├── convert_turbo_vaed.py      — Turbo-VAED decoder export
│   │   ├── convert_vae_encoder.py     — VAE encoder export
│   │   ├── requirements.txt           — Python dependencies (torch, onnx, transformers, etc.)
│   │   └── README.md                  — Conversion pipeline documentation
│   ├── model/                         — Model inspection utilities
│   │   ├── cache/                     — Downloaded model files
│   │   │   ├── models--Lightricks--LTX-Video/
│   │   │   ├── models--Qwen--Qwen2-0.5B/
│   │   │   ├── turbo_vaed/
│   │   │   └── hybrid_371.pth
│   │   ├── inspect_checkpoint.py      — Basic checkpoint inspection
│   │   └── inspect_detailed.py        — Detailed model structure dump
│   └── notebooks/                     — Jupyter/Colab notebooks
│       └── MobileI2V_ONNX_Converter.ipynb
├── src/                               — .NET MAUI Android application
│   ├── MobileI2VConsole/              — Main MAUI app project
│   │   ├── Converters/                — XAML value converters
│   │   │   ├── InverseBoolConverter.cs
│   │   │   └── NotNullToBoolConverter.cs
│   │   ├── Models/                    — Data models (C# records)
│   │   │   ├── GenerationProgress.cs  — Pipeline step progress info
│   │   │   ├── GenerationRequest.cs   — Input params (image path, prompt, dimensions)
│   │   │   ├── GenerationResult.cs    — Output (video path, elapsed, success)
│   │   │   ├── ModelStatus.cs         — ONNX model download/load status
│   │   │   └── PromptTemplate.cs      — Predefined text prompt templates
│   │   ├── Services/                  — Service interfaces & implementations
│   │   │   ├── IFileService.cs        — File operations interface
│   │   │   ├── FileService.cs         — File service implementation
│   │   │   ├── IInferenceOrchestrator.cs — Full I2V pipeline orchestrator
│   │   │   ├── InferenceOrchestrator.cs  — Pipeline orchestrator impl
│   │   │   ├── IMediaPickerService.cs — Image selection interface
│   │   │   ├── MediaPickerService.cs  — Media picker implementation
│   │   │   ├── IModelManager.cs       — ONNX model management interface
│   │   │   ├── ModelManagerService.cs — Model download & session management
│   │   │   ├── IVideoEncoder.cs       — Video encoding interface
│   │   │   └── StubVideoEncoder.cs    — Fallback stub (non-Android)
│   │   ├── ViewModels/                — MVVM ViewModels
│   │   │   ├── HomeViewModel.cs       — Main/home screen logic
│   │   │   ├── GenerationViewModel.cs — Generation progress & controls
│   │   │   └── ResultViewModel.cs     — Result display & sharing
│   │   ├── Views/                     — XAML UI pages
│   │   │   ├── HomePage.xaml / .cs     — Image picker & prompt input
│   │   │   ├── GenerationPage.xaml / .cs — Generation progress UI
│   │   │   └── ResultPage.xaml / .cs   — Video playback & share
│   │   ├── Platforms/Android/         — Android platform-specific code
│   │   │   ├── AndroidManifest.xml
│   │   │   ├── AndroidVideoEncoder.cs — Hardware H.264 encoder (Android)
│   │   │   ├── MainActivity.cs        — Android activity
│   │   │   └── MainApplication.cs     — Android app class
│   │   ├── Resources/                 — App resources
│   │   │   ├── AppIcon/appicon.svg
│   │   │   ├── Splash/splash.svg
│   │   │   ├── Raw/                   — Bundled ONNX models + prompt templates
│   │   │   │   ├── mobilei2v_unet.onnx
│   │   │   │   ├── qwen2_encoder.onnx
│   │   │   │   ├── qwen2_encoder.onnx.data
│   │   │   │   ├── turbo_vaed.onnx
│   │   │   │   ├── vae_encoder.onnx
│   │   │   │   └── PromptTemplates.json
│   │   │   └── Styles/                — XAML styles & colors
│   │   │       ├── Colors.xaml
│   │   │       └── Styles.xaml
│   │   ├── App.xaml / .xaml.cs        — Application entry point
│   │   ├── AppShell.xaml / .xaml.cs   — Shell navigation
│   │   ├── MauiProgram.cs             — DI registration & MAUI setup
│   │   └── MobileI2VConsole.csproj    — .NET project (net10.0-android)
│   └── skills/                        — Opencode/FlowDeck agent skills
│       └── pytorch-onnx-debug/SKILL.md
├── tests/                             — .NET test project
│   └── MobileI2VConsole.Tests/        — xUnit test project
│       ├── Models/
│       │   └── ModelRecordTests.cs    — Tests for model record types
│       ├── Services/
│       │   ├── FileServiceTests.cs
│       │   └── ModelManagerServiceTests.cs
│       ├── ViewModels/
│       │   └── HomeViewModelTests.cs
│       ├── MobileI2VConsole.Tests.csproj — Test project (net10.0-android, xUnit + Moq)
│       ├── bin/                       — Build output (gitignored)
│       └── obj/                       — Intermediate objects (gitignored)
├── MobileI2VConsole.sln               — Visual Studio solution file
├── mobilei2v_unet.onnx                — Root-level exported UNet model
└── README.md                          — Project overview
```

## File Naming Conventions

| Language | Pattern | Examples |
|----------|---------|---------|
| **C#** | `PascalCase.cs` | `App.xaml.cs`, `HomeViewModel.cs`, `FileService.cs`, `InverseBoolConverter.cs` |
| **C# tests** | `PascalCaseTests.cs` | `HomeViewModelTests.cs`, `FileServiceTests.cs` |
| **Python** | `snake_case.py` | `model_utils.py`, `ort_config.py`, `convert_mobilei2v_unet.py` |
| **Python tests** | `test_snake_case.py` | `test_turbo_vaed_bug.py` |
| **XAML** | `PascalCase.xaml` | `HomePage.xaml`, `Colors.xaml` |
| **Config/data** | `PascalCase.json` | `PromptTemplates.json` |
| **Markdown** | `UPPER_CASE.md` | `README.md` |
| **VS Solution** | `PascalCase.sln` | `MobileI2VConsole.sln` |

## Module Organization

### C# MAUI App (`src/MobileI2VConsole/`)

- **Namespaces** match the folder hierarchy: `MobileI2VConsole.Models`, `MobileI2VConsole.Services`, `MobileI2VConsole.ViewModels`, etc.
- **Dependency Injection** is wired in `MauiProgram.cs` using `AddSingleton` / `AddTransient`:
  - Services registered as singleton: `IFileService`, `IModelManager`, `IMediaPickerService`
  - Services registered as transient: `IInferenceOrchestrator`
  - ViewModels and Pages registered as transient
  - `IVideoEncoder` resolved conditionally: Android uses `AndroidVideoEncoder`, others use `StubVideoEncoder`
- **MVVM pattern** with CommunityToolkit.Mvvm (`ObservableObject`, `RelayCommand`)
- **View-to-ViewModel binding** set in XAML code-behind constructors

### Python Conversion Scripts (`scripts/convert/`)

- **Package structure** uses `__init__.py` files in `common/`, `models/`, and `tests/`
- **Model definitions** are self-contained: `mobiledit.py` vendors all dependencies (timm helpers, einops patterns) inline — no external model package dependency
- **ONNX export scripts** are standalone top-level scripts, not part of a package
- **Shared utilities** in `common/` package: `model_utils.py` (device, download, export), `ort_config.py` (optimization, FP16)

### C# Test Project (`tests/MobileI2VConsole.Tests/`)

- **Mirrors** the `src/MobileI2VConsole/` project structure under `Models/`, `Services/`, `ViewModels/`
- **Uses xUnit** as the test framework with `Moq` for mocking
- **References** the main project via `ProjectReference`

## Configuration Files

| File | Purpose |
|------|---------|
| `MobileI2VConsole.sln` | Visual Studio solution file referencing `src/MobileI2VConsole/MobileI2VConsole.csproj` and `tests/MobileI2VConsole.Tests/MobileI2VConsole.Tests.csproj` |
| `src/MobileI2VConsole/MobileI2VConsole.csproj` | .NET project: targets `net10.0-android`, enables MAUI, references ONNX Runtime (1.27.0), CommunityToolkit.Mvvm (8.4.2), CommunityToolkit.Maui (14.2.0), SkiaSharp (4.148.0) |
| `tests/MobileI2VConsole.Tests/MobileI2VConsole.Tests.csproj` | Test project: targets `net10.0-android`, references xUnit (2.9.3), Moq (4.20.72), coverlet (6.0.4) |
| `src/MobileI2VConsole/Platforms/Android/AndroidManifest.xml` | Android permissions and app manifest |
| `scripts/convert/requirements.txt` | Python deps: torch≥2.1.0, onnx≥1.15.0, transformers≥4.44.0, diffusers≥0.30.0, etc. |
| `src/MobileI2VConsole/Resources/Raw/PromptTemplates.json` | Predefined text prompt templates for UI chip buttons |
| `src/MobileI2VConsole/Resources/Styles/Colors.xaml` | XAML color palette definitions |
| `src/MobileI2VConsole/Resources/Styles/Styles.xaml` | XAML style resources for controls |
| `.gitignore` | Ignores build artifacts (bin/obj), ONNX files at root and in model dirs, VS settings, Python cache |
| `src/skills/pytorch-onnx-debug/SKILL.md` | Opencode/FlowDeck skill for PyTorch→ONNX debugging |

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/MobileI2VConsole/MauiProgram.cs` | DI container setup, service registration, MAUI app builder |
| `src/MobileI2VConsole/App.xaml.cs` | Application entry — creates window with `AppShell` |
| `src/MobileI2VConsole/AppShell.xaml` | Shell-based navigation structure |
| `src/MobileI2VConsole/ViewModels/HomeViewModel.cs` | Home screen: image picking, prompt input, navigation to generation |
| `src/MobileI2VConsole/ViewModels/GenerationViewModel.cs` | Generation pipeline: model loading, inference, progress reporting |
| `src/MobileI2VConsole/ViewModels/ResultViewModel.cs` | Result screen: video playback, share, retry |
| `src/MobileI2VConsole/Services/IInferenceOrchestrator.cs` | Interface defining the full I2V pipeline contract |
| `src/MobileI2VConsole/Services/InferenceOrchestrator.cs` | Orchestrates: VAE encode → Qwen2 encode → UNet denoise → Turbo-VAED decode |
| `src/MobileI2VConsole/Services/ModelManagerService.cs` | Downloads, caches, and loads ONNX models into ORT sessions |
| `src/MobileI2VConsole/Platforms/Android/AndroidVideoEncoder.cs` | Android hardware-accelerated H.264 video encoding |
| `scripts/convert/models/mobiledit.py` | Complete MobileI2V UNet architecture (vendored, 1400+ lines) |
| `scripts/convert/common/model_utils.py` | Shared ONNX export utilities (device, download, verify) |
| `scripts/convert/common/ort_config.py` | ONNX Runtime session configuration and optimization |
| `scripts/convert/convert_all.py` | Batch entry point for converting all 4 models to ONNX |

## Empty/Potentially Stale Directories

- `src/MobileI2VConsole/Resources/Images/` and `Resources/Fonts/` are referenced in the `.csproj` (`MauiImage`, `MauiFont`) but the directories were not present on disk — they may not exist yet or were cleaned.

## Build Artifacts (gitignored)

- `src/MobileI2VConsole/bin/`, `src/MobileI2VConsole/obj/` — .NET build output
- `tests/MobileI2VConsole.Tests/bin/`, `tests/MobileI2VConsole.Tests/obj/` — Test build output
- `scripts/convert/models/*.onnx`, `scripts/convert/models/*.onnx.data` — Generated ONNX files
- `scripts/convert/*/__pycache__/` — Python bytecode cache
- `scripts/model/*` — Model checkpoint cache
- `mobilei2v_unet.onnx` (root) — Exported ONNX model (built from convert script)
