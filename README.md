# MobileI2V Console

**On-device image-to-video generation for Android.** Converts static images into 720p H.264 MP4 video using the MobileI2V diffusion model and Turbo-VAED video decoder, running entirely on-device via ONNX Runtime.

[![Target](https://img.shields.io/badge/Android-8.0%2B-3DDC84?logo=android)](https://developer.android.com)
[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4?logo=dotnet)](https://dotnet.microsoft.com)
[![MAUI](https://img.shields.io/badge/MAUI-10.0-512BD4?logo=dotnet)](https://learn.microsoft.com/dotnet/maui)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.27.0-005CED?logo=onnx)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

MobileI2V Console brings the [MobileI2V](https://huggingface.co/hustvl/MobileI2V) image-to-video diffusion model to Android devices. It combines a lightweight UNet with a 2-step turbo diffusion schedule and the Turbo-VAED video decoder to produce short 720p video clips from a single image and optional text prompt.

**Key features:**

- **Fully on-device** — no cloud API calls. All inference runs locally via ONNX Runtime.
- **Image-to-video** — pick an image from your gallery or camera, optionally describe the desired motion with text.
- **720p output** — 17 frames at 17 FPS, encoded to H.264 MP4 via Android MediaCodec.
- **Prompt templates** — quick-select motion descriptions (gentle motion, zoom, pan, etc.).
- **MVVM architecture** — built with .NET MAUI and CommunityToolkit.Mvvm.
- **Cancelable pipeline** — progress reporting and cancellation at every step.

---

## Architecture

The app follows the MVVM pattern with dependency injection. The inference pipeline runs four ONNX models in sequence, then encodes the output frames to video.

```
┌─────────────────────────────────────────────────────────────────┐
│                         MobileI2V Console                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐               │
│  │ HomePage  │───▶│GenerationPage│───▶│ResultPage│               │
│  │ (input)   │    │ (progress)   │    │ (preview)│               │
│  └──────────┘    └──────────────┘    └──────────┘               │
│       │                 │                    │                    │
│  ┌────▼────────────────▼────────────────────▼───┐               │
│  │           ViewModels (MVVM)                   │               │
│  └────▲────────────────▲────────────────────▲───┘               │
│       │                 │                    │                    │
│  ┌────▼────────────────▼────────────────────▼───┐               │
│  │        Services (DI Container)                │               │
│  │ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │               │
│  │ │ModelMgr  │ │FileSvc   │ │InferenceOrch   │ │               │
│  │ │(download)│ │(storage) │ │(pipeline)      │ │               │
│  │ └──────────┘ └──────────┘ └───────┬────────┘ │               │
│  │ ┌──────────┐ ┌──────────┐         │          │               │
│  │ │MediaPckr │ │VideoEnc  │◀────────┘          │               │
│  │ │(gallery) │ │(H.264)   │                    │               │
│  │ └──────────┘ └──────────┘                    │               │
│  └──────────────────────────────────────────────┘               │
│                        │                                         │
│  ┌─────────────────────▼───────────────────────────────────────┐ │
│  │              ONNX Runtime Inference Pipeline                  │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │ │
│  │  │VAE Enc.  │  │Qwen2-0.5B│  │MobileI2V │  │Turbo-VAED   │ │ │
│  │  │img→latent│  │txt→emb   │  │UNet      │  │Dec. latent→ │ │ │
│  │  │  ~50MB   │  │  ~900MB  │  │denoise   │  │frame        │ │ │
│  │  │          │  │          │  │  ~540MB  │  │  ~50MB      │ │ │
│  │  └──────────┘  └──────────┘  └────┬─────┘  └─────────────┘ │ │
│  │                                   │                         │ │
│  │                                   ▼ 17 latent frames        │ │
│  │                          ┌──────────────────┐               │ │
│  │                          │Android MediaCodec│               │ │
│  │                          │H.264 → MP4 mux   │               │ │
│  │                          └──────────────────┘               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline (6 steps)

| Step | Model | Input | Output | Purpose |
|------|-------|-------|--------|---------|
| 1 | VAE Encoder | Image (720p RGBA) | 4×64×64 latent | Compress image to latent space |
| 2 | Qwen2-0.5B Encoder | Text prompt (tokens) | 768-d embeddings | Encode text conditioning |
| 3 | MobileI2V UNet (step 1) | Latent + embeddings | Denoised latent | 1st diffusion denoising step |
| 4 | MobileI2V UNet (step 2) | Latent + embeddings | 17 latent frames | 2nd denoising step (turbo shortcut) |
| 5 | Turbo-VAED Decoder | Per-frame latent | 1280×720 RGBA (×17) | Decode each latent to video frame |
| 6 | Android MediaCodec | 17 RGBA frames | H.264 MP4 @ 17 FPS | Encode and mux to video file |

The UNet uses a 2-step turbo diffusion schedule (instead of the usual 25-50 steps), making it feasible for on-device inference while retaining reasonable quality.

---

## Prerequisites

### Build Requirements

- [.NET SDK 10.0](https://dotnet.microsoft.com/download/dotnet/10.0) or later
- Android SDK (API 26+), installed via Visual Studio or the [Android SDK Manager](https://developer.android.com/studio/command-line/sdkmanager)
- For Windows: Visual Studio 2022+ with the **Mobile development with .NET** workload
- For macOS: Visual Studio for Mac or `dotnet` CLI with Android workloads

### Model Conversion Requirements

- Python 3.10 or later
- See `scripts/convert/requirements.txt` for Python dependencies

### Runtime Requirements

- Android device or emulator running **Android 8.0 (API 26) or later**
- ~2.5 GB of free storage for downloaded models
- ~2 GB of RAM recommended (model inference is memory-intensive)

---

## Setup & Build

### 1. Clone the repository

```bash
git clone <repo-url>
cd MobileI2VConsole
```

### 2. Restore NuGet packages

```bash
dotnet restore
```

### 3. Build the Android app

```bash
dotnet build -f net10.0-android
```

The APK/AAB will be produced in `src/MobileI2VConsole/bin/Release/net10.0-android/`.

> **Note:** The first build may take several minutes as NuGet restores and the .NET MAUI Android workload is prepared. Subsequent builds are faster.

### 4. Convert models to ONNX (required before first run)

The app needs four ONNX models downloaded from HuggingFace and converted. Run the conversion script:

```bash
# Set up Python environment
cd scripts/convert
python -m venv venv
source venv/bin/activate    # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run conversion (downloads models from HuggingFace and exports to ONNX)
python convert_all.py --output ../../src/MobileI2VConsole/Resources/Raw/models/

# For mobile-optimized ORT format (smaller, faster):
python convert_all.py --output ../../src/MobileI2VConsole/Resources/Raw/models/ --optimize
```

This script creates a `models_manifest.json` alongside the ONNX files. The app reads this manifest on startup to locate and verify the models.

**What this does:**

| Step | Action |
|------|--------|
| Phase 1 | Downloads VAE Encoder from HuggingFace, exports to ONNX |
| Phase 2 | Downloads Qwen2-0.5B text encoder, exports to ONNX |
| Phase 3 | Downloads MobileI2V UNet, exports to ONNX |
| Phase 4 | Downloads Turbo-VAED Decoder, exports to ONNX |
| Post | Generates `models_manifest.json` with SHA256 hashes and file sizes |

> **Note:** The Qwen2-0.5B encoder is ~900 MB. Download time depends on your internet connection. The `--optimize` flag additionally produces ORT-format models for reduced size and faster load times.

---

## Running the App

### From the CLI

```bash
dotnet build -t:Run -f net10.0-android
```

This builds and deploys the app to a connected device or running emulator. The `-t:Run` flag launches the app automatically.

### From Visual Studio

1. Open `MobileI2VConsole.sln`
2. Set the target to **Android** (not Windows or any other framework)
3. Select a connected device or Android emulator from the run dropdown
4. Press **F5** to build and deploy

### Manual APK deployment

```bash
# Build the APK
dotnet publish -f net10.0-android -c Release

# Install on connected device
adb install src/MobileI2VConsole/bin/Release/net10.0-android/*-Signed.apk
```

### First Launch

When the app starts for the first time, it checks for the presence of ONNX models in the app's data directory:

1. **Model download screen** — If models are missing (e.g., on first install), the app shows a download button. The models are bundled in `Resources/Raw/models/` from the conversion step above.
2. **Grant permissions** — The app requests gallery, camera, and storage permissions as needed.
3. **Pick an image** — Tap the image area to select from gallery or camera.
4. **Enter a prompt (optional)** — Describe the desired motion, or use one of the template chips.
5. **Generate** — Tap the generate button. The pipeline takes approximately 10-30 seconds depending on device.

---

## Project Layout

```
MobileI2VConsole/
├── MobileI2VConsole.sln                          # Solution file
│
├── src/MobileI2VConsole/                         # Main app project
│   ├── MobileI2VConsole.csproj                   # Project file (.NET MAUI Android)
│   ├── MauiProgram.cs                            # DI container & app builder
│   ├── App.xaml / App.xaml.cs                     # Application entry point
│   ├── AppShell.xaml / AppShell.xaml.cs           # Shell navigation
│   │
│   ├── Models/                                   # Data contracts
│   │   ├── GenerationRequest.cs                  #   Input: image, prompt, resolution
│   │   ├── GenerationResult.cs                   #   Output: video path, status
│   │   ├── GenerationProgress.cs                 #   Pipeline progress (step, frame)
│   │   ├── ModelStatus.cs                        #   Download/load status per model
│   │   └── PromptTemplate.cs                     #   Predefined prompt chip data
│   │
│   ├── Services/                                 # Business logic layer
│   │   ├── IFileService.cs / FileService.cs      #   File I/O & app data directory
│   │   ├── IModelManager.cs / ModelManagerService.cs    # Model download & loading
│   │   ├── IMediaPickerService.cs / MediaPickerService.cs  # Gallery & camera access
│   │   ├── IInferenceOrchestrator.cs / InferenceOrchestrator.cs  # Full pipeline orchestration
│   │   └── IVideoEncoder.cs                      #   Video encoding abstraction
│   │       ├── AndroidVideoEncoder.cs            #   Android MediaCodec H.264 encoder
│   │       └── StubVideoEncoder.cs               #   Stub for non-Android targets
│   │
│   ├── ViewModels/                               # MVVM ViewModels
│   │   ├── HomeViewModel.cs                      #   Image picker, prompt, launch
│   │   ├── GenerationViewModel.cs                #   Pipeline progress & cancel
│   │   └── ResultViewModel.cs                    #   Video preview & share
│   │
│   ├── Views/                                    # XAML pages
│   │   ├── HomePage.xaml / .cs                   #   Image + prompt input
│   │   ├── GenerationPage.xaml / .cs             #   Progress indicators
│   │   └── ResultPage.xaml / .cs                 #   Video playback
│   │
│   ├── Converters/                               # XAML value converters
│   │   ├── InverseBoolConverter.cs
│   │   └── NotNullToBoolConverter.cs
│   │
│   ├── Resources/                                # App resources
│   │   ├── AppIcon/                              #   App launcher icon
│   │   ├── Splash/                               #   Splash screen
│   │   ├── Images/                               #   Embedded images
│   │   ├── Fonts/                                #   OpenSans fonts
│   │   └── Raw/models/                           #   ONNX model files (after conversion)
│   │
│   └── Platforms/Android/                        # Android-specific code
│       ├── AndroidManifest.xml                   #   Permissions & manifest config
│       ├── MainActivity.cs                       #   Android activity entry point
│       ├── MainApplication.cs                    #   Android application class
│       └── AndroidVideoEncoder.cs                #   MediaCodec H.264 → MP4 encoder
│
├── scripts/convert/                              # Python ONNX conversion pipeline
│   ├── requirements.txt                          #   Python dependencies
│   ├── convert_all.py                            #   Orchestrator: converts all 4 models
│   ├── convert_vae_encoder.py                    #   VAE Encoder → ONNX
│   ├── convert_qwen2_encoder.py                  #   Qwen2-0.5B → ONNX
│   ├── convert_mobilei2v_unet.py                 #   MobileI2V UNet → ONNX
│   ├── convert_turbo_vaed.py                     #   Turbo-VAED Decoder → ONNX
│   ├── common/                                   #   Shared conversion utilities
│   ├── models/                                   #   Local model cache during conversion
│   └── README.md                                 #   Conversion-specific docs
│
└── tests/
    └── MobileI2VConsole.Tests/                   # xUnit test project
        ├── MobileI2VConsole.Tests.csproj         #   Test project (xUnit + Moq)
        └── ...                                   #   Test files
```

---

## Screens

### HomePage

- **Image picker** — Two buttons: Gallery (opening the system photo picker) and Camera (capturing a new photo). Selected image is previewed in a rounded frame.
- **Prompt input** — Text entry field for describing the desired motion (e.g., "waves crashing", "clouds drifting").
- **Prompt templates** — Horizontal scrollable chips with common motion descriptions. Tapping a chip fills the prompt field.
- **Model status** — Shows download progress and a "Download Models" button if models are not yet present.
- **Generate button** — Launches the generation pipeline.

### GenerationPage

- **Step indicators** — Shows the current pipeline step (encoding, denoising, decoding, encoding video).
- **Progress bar** — Overall progress across all pipeline steps.
- **Frame counter** — Current frame being decoded / total frames.
- **Cancel button** — Aborts the pipeline and returns to HomePage.

### ResultPage

- **Video preview** — Plays the generated MP4 using `MediaElement`.
- **Share button** — Shares the video via the Android share sheet.
- **Save button** — Saves the video to the device gallery.
- **Generate new** — Returns to HomePage to start again.

---

## Pipeline Detail

### Step 1: VAE Encoder

The VAE encoder compresses the 1280×720 RGBA input image into a 4-channel 64×64 latent representation. This reduces the data volume by ~99.9% before the diffusion process.

```csharp
// Conceptual: image → latent tensor
float[] latent = vaeEncoder.Run(imageTensor);  // shape: [1, 4, 64, 64]
```

### Step 2: Qwen2-0.5B Text Encoder

The text prompt is tokenized and passed through the Qwen2-0.5B transformer to produce a 768-dimensional embedding vector that conditions the diffusion process. If no prompt is provided, an empty/null embedding is used.

```csharp
float[] textEmbedding = qwen2Encoder.Run(tokenizedPrompt);  // shape: [1, 768]
```

### Step 3-4: MobileI2V UNet (2-step turbo diffusion)

The UNet performs two denoising steps (a "turbo" shortcut vs. the typical 25-50 steps):

1. **Step 1:** Takes the image latent + text embedding + noise, produces a partially denoised latent.
2. **Step 2:** Takes the step 1 output + text embedding, produces the final 17-frame latent tensor.

The output is a tensor of shape `[17, 4, 64, 64]` — 17 latent frames, each 4×64×64.

```csharp
float[] noisyLatent = AddNoise(imageLatent);
float[] step1 = mobileI2VUnet.Run(noisyLatent, textEmbedding);     // step 1
float[] step2 = mobileI2VUnet.Run(step1, textEmbedding);           // step 2
// step2 shape: [17, 4, 64, 64] — 17 latent frames
```

### Step 5: Turbo-VAED Decoder

Each of the 17 latent frames is decoded from latent space to a full 1280×720 RGBA image. The Turbo-VAED decoder is optimized for video and processes all 17 frames sequentially.

```csharp
byte[][] frames = new byte[17][];
for (int i = 0; i < 17; i++)
{
    frames[i] = turboVaedDecoder.Run(step2[i]);  // [4,64,64] → [1280*720*4] RGBA
}
```

### Step 6: Android MediaCodec H.264 Encoder

The raw RGBA frames are converted to NV12 (YUV420 semi-planar) format and fed to Android's `MediaCodec` H.264 encoder. The encoder muxes the output into an MP4 container at 17 FPS with a 4 Mbps bitrate.

```csharp
string videoPath = await videoEncoder.EncodeFramesAsync(
    frames, outputPath, width: 1280, height: 720, fps: 17);
```

---

## Models

All models are sourced from HuggingFace and converted to ONNX format via the Python scripts in `scripts/convert/`.

| Model | Source | Format | Size | Input | Output |
|-------|--------|--------|------|-------|--------|
| VAE Encoder | [hustvl/MobileI2V](https://huggingface.co/hustvl/MobileI2V) | ONNX | ~50 MB | 1280×720 RGBA image | 4×64×64 latent |
| Qwen2-0.5B Encoder | [hustvl/MobileI2V](https://huggingface.co/hustvl/MobileI2V) | ONNX | ~900 MB | Tokenized text | 768-d embedding |
| MobileI2V UNet | [hustvl/MobileI2V](https://huggingface.co/hustvl/MobileI2V) | ONNX | ~540 MB | Latent + embedding + noise | 17×4×64×64 latent |
| Turbo-VAED Decoder | [hustvl/Turbo-VAED](https://huggingface.co/hustvl/Turbo-VAED) | ONNX | ~50 MB | 4×64×64 latent | 1280×720 RGBA |

**Total model storage:** ~1.5 GB (ONNX) or ~1.2 GB (ORT-optimized).

---

## Android Permissions

The following permissions are declared in `AndroidManifest.xml`:

| Permission | Purpose | Required When |
|------------|---------|---------------|
| `INTERNET` | Model download from HuggingFace | First launch / model update |
| `ACCESS_NETWORK_STATE` | Network availability check | First launch / model update |
| `READ_MEDIA_IMAGES` | Gallery image picker | Android 13+ (API 33+) |
| `READ_EXTERNAL_STORAGE` | Gallery image picker | Android 8-12 (API 26-32) |
| `CAMERA` | Photo capture | When using camera input |
| `WRITE_EXTERNAL_STORAGE` | Save video to gallery | When saving result |

On Android 13+ (API 33+), the system requests `READ_MEDIA_IMAGES` at runtime. On older versions, `READ_EXTERNAL_STORAGE` is requested instead. The app uses the modern photo picker where available, which may not require storage permissions at all.

---

## Development

### Adding a new model

1. **Write a converter** — Create a Python script in `scripts/convert/` (e.g., `convert_my_model.py`) that downloads the model from HuggingFace and exports it to ONNX.
2. **Register in the pipeline** — Add the converter to `convert_all.py` and update `ModelManagerService.cs` to load the new ONNX model.
3. **Add to manifest** — Update `GenerationProgress.cs` if the model adds new pipeline steps, and update the progress UI in `GenerationPage.xaml`.

### Running tests

```bash
dotnet test tests/MobileI2VConsole.Tests
```

Tests use **xUnit** with **Moq** for service mocking and **coverlet** for code coverage.

```bash
# With coverage report
dotnet test tests/MobileI2VConsole.Tests /p:CollectCoverage=true
```

### Code style

- The project uses nullable reference types (`<Nullable>enable</Nullable>` in csproj).
- All public services use interface/implementation separation (`IFoo` + `FooService`).
- ViewModels use `CommunityToolkit.Mvvm` source generators (`[ObservableProperty]`, `[RelayCommand]`).
- DI registration is centralized in `MauiProgram.cs`.

---

## Troubleshooting

### Build errors

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `NETSDK1136` `The target framework 'net10.0-android' was not found` | .NET 10 SDK not installed | Install [.NET 10.0 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) |
| `APPX0002` `The 'Android' target platform is not installed` | Android workload missing | `dotnet workload install maui-android` or install via Visual Studio installer |
| `XA0000` `Could not find android.jar` | Android SDK not found | Set `ANDROID_HOME` environment variable to your Android SDK path |
| Build succeeds but deployment fails | Device not detected or API level too low | Verify `adb devices` lists your device. Ensure device runs API 26+. |

### Model conversion errors

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `OSError: [E050] Can't find model` | HuggingFace model not cached | Check internet connection. Models download automatically on first run. |
| `CUDA out of memory` | GPU VRAM exhausted during conversion | Use `--device cpu` to convert on CPU (slower but works) |
| `onnx.onnx_cpp2py_export.mapper.MappingError` | ONNX opset version mismatch | Update `torch` and `onnx` to latest versions: `pip install --upgrade torch onnx` |
| `TypeError: 'NoneType' object is not subscriptable` | Model name mismatch | Verify the model name in the converter script matches the HuggingFace repo |

### Runtime errors

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `OnnxRuntimeException` `Model not found` | ONNX models not in app data directory | Run `convert_all.py` with output pointing to `Resources/Raw/models/` and rebuild |
| `MediaCodec` encoder creation fails | Device H.264 encoder not available | Check device supports H.264 encoding. Rare on modern Android devices. |
| App crashes during model load | Out of memory | Close other apps. Some devices with <2 GB RAM may not have enough contiguous memory. |
| Video appears corrupted or green | NV12 conversion issue | Verify `ConvertRgbaToNv12` in `AndroidVideoEncoder.cs` matches the device's expected YUV format |

### Permission issues

- On Android 13+ (API 33+), gallery access uses `READ_MEDIA_IMAGES`. The `READ_EXTERNAL_STORAGE` permission is not requested on these devices.
- Camera permission is only requested when the user taps the camera button.
- If permissions are denied, the app gracefully falls back to the other input method.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- **[MobileI2V](https://huggingface.co/hustvl/MobileI2V)** — The image-to-video diffusion model (HustVL / Huazhong University of Science and Technology).
- **[Turbo-VAED](https://huggingface.co/hustvl/Turbo-VAED)** — The efficient video decoder for latent-to-frame conversion.
- **[ONNX Runtime](https://onnxruntime.ai)** — Cross-platform inference engine for ONNX models.
- **[.NET MAUI](https://learn.microsoft.com/dotnet/maui)** — Cross-platform UI framework for building native mobile apps with .NET.
- **[CommunityToolkit.Mvvm](https://learn.microsoft.com/dotnet/communitytoolkit/mvvm)** — MVVM source generators and observable patterns.
- **[SkiaSharp](https://github.com/mono/SkiaSharp)** — Cross-platform 2D graphics library.
