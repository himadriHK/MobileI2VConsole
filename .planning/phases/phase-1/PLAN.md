# Phase 1 — Implementation Plan: MobileI2V Console

## Overview

MAUI Android app converting images to 720p video using on-device MobileI2V + Turbo-VAED models.

---

## 3 Delivery Waves

### Wave 1 — Foundation (All Parallel)

**WS-A: ONNX Conversion Scripts** (scripts/convert/)
- `convert_vae_encoder.py` — Export VAE encoder (image → latent)
- `convert_qwen2_encoder.py` — Export Qwen2-0.5B text encoder → ONNX
- `convert_mobilei2v_unet.py` — Export MobileI2V denoising UNet → ONNX
- `convert_turbo_vaed.py` — Export Turbo-VAED decoder → ONNX
- `convert_all.py` — Orchestrator: downloads weights, runs all conversions, validates outputs
- `requirements.txt` — Python dependencies
- `README.md` — Usage instructions
- **Verification**: Each script produces a valid ONNX file loadable by onnxruntime

**WS-B: MAUI Solution Scaffold**
- `MobileI2VConsole.sln` — Solution file
- `src/MobileI2VConsole/MobileI2VConsole.csproj` — Project with NuGet refs
- `src/MobileI2VConsole/App.xaml` + `.cs` — Application entry
- `src/MobileI2VConsole/AppShell.xaml` + `.cs` — Shell navigation (3 routes)
- `src/MobileI2VConsole/Platforms/Android/AndroidManifest.xml` — Permissions
- `src/MobileI2VConsole/Platforms/Android/MainActivity.cs` — Main activity
- `src/MobileI2VConsole/Platforms/Android/MainApplication.cs` — App class
- `src/MobileI2VConsole/MauiProgram.cs` — DI registration
- `src/MobileI2VConsole/Resources/AppIcon/appicon.svg` — App icon
- **Verification**: `dotnet build` succeeds for net8.0-android

**WS-C: Data Models** (src/MobileI2VConsole/Models/)
- `GenerationRequest.cs` — Input image path + prompt + settings
- `GenerationResult.cs` — Output video path + metadata
- `ModelStatus.cs` — Download status enum + progress
- `GenerationProgress.cs` — Step name + percentage + current frame
- `PromptTemplate.cs` — Name + prompt text
- **Verification**: All records compile

**WS-D: Service Interfaces** (src/MobileI2VConsole/Services/)
- `IModelManager.cs` — DownloadModels(), LoadModel(), UnloadModel(), GetStatus()
- `IInferenceOrchestrator.cs` — GenerateVideoAsync(request, progress) → Result
- `IVideoEncoder.cs` — EncodeFramesAsync(frames, outputPath, progress) → string
- `IMediaPickerService.cs` — PickImageAsync() → FileResult, CaptureImageAsync() → FileResult
- `IFileService.cs` — GetCacheDir(), GetOutputDir(), CleanupAsync()
- **Verification**: All interfaces compile

**WS-E: Resources**
- `Resources/Styles/Colors.xaml`
- `Resources/Styles/Styles.xaml`
- `Resources/Raw/PromptTemplates.json` — 6 default templates
- **Verification**: XAML resources load at runtime

### Wave 2 — Core Services (Sequential: F → G → H)

**WS-F: File & Model Services**
- `Services/FileService.cs` — Manages app cache/output directories
- `Services/ModelManagerService.cs` — HuggingFace download (HTTP Range resume, SHA256), ONNX session lifecycle
- **Deps**: WS-C, WS-D
- **Verification**: Model downloads + loads successfully

**WS-G: Media & Inference Services**
- `Services/MediaPickerService.cs` — Wraps MAUI MediaPicker (gallery + camera)
- `Services/InferenceOrchestrator.cs` — Full pipeline: load models → VAE encode → Qwen encode → UNet 2-step denoising → Turbo-VAED decode → raw frames
- **Deps**: WS-F (for model loading)
- **Verification**: Orchestrator produces 17 raw frame byte arrays

**WS-H: Video Encoder**
- `Services/AndroidVideoEncoder.cs` — Android MediaCodec H.264 + MediaMuxer MP4
- **Deps**: WS-G (raw frames produced)
- **Verification**: Produces valid .mp4 playable on device

### Wave 3 — UI + Tests (Parallel after Wave 2)

**WS-I: ViewModels**
- `ViewModels/HomeViewModel.cs` — Image pick + prompt + navigate to generation
- `ViewModels/GenerationViewModel.cs` — Progress reporting, pipeline orchestration
- `ViewModels/ResultViewModel.cs` — Video playback, share, save to gallery
- **Deps**: WS-F, WS-G, WS-H (services)
- **Verification**: All commands work end-to-end

**WS-J: XAML Pages**
- `Views/HomePage.xaml` + `.cs` — Image preview, prompt input with template chips, generate button, recent results strip
- `Views/GenerationPage.xaml` + `.cs` — Step progress indicators, cancel button
- `Views/ResultPage.xaml` + `.cs` — MediaElement player, share/save buttons
- **Deps**: WS-I (ViewModels)
- **Verification**: All pages navigate and render

**Tests**
- `tests/MobileI2VConsole.Tests/Services/ModelManagerServiceTests.cs`
- `tests/MobileI2VConsole.Tests/Services/InferenceOrchestratorTests.cs`
- `tests/MobileI2VConsole.Tests/Services/VideoEncoderTests.cs`
- `tests/MobileI2VConsole.Tests/ViewModels/HomeViewModelTests.cs`
- `tests/MobileI2VConsole.Tests/ViewModels/GenerationViewModelTests.cs`
- `tests/MobileI2VConsole.Tests/ViewModels/ResultViewModelTests.cs`

---

## Dependency Graph

```
Wave 1 (parallel):
  [WS-A] [WS-B] [WS-C] [WS-D] [WS-E]

Wave 2 (sequential):
  WS-F ──→ WS-G ──→ WS-H

Wave 3 (parallel):
  WS-I ──┐
  WS-J ──┼──→ Tests
         │
  ───────┘
```

## Verification Criteria

1. Python scripts produce valid ONNX files
2. `dotnet build` succeeds for net8.0-android
3. Model download + ONNX session load works on device/emulator
4. Inference pipeline produces 17 raw frames
5. Video encoder produces valid .mp4
6. 3-screen UI navigates correctly
7. All unit tests pass
