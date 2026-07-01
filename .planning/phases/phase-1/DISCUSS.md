# Phase 1 — Requirements Discussion

## Decisions

| ID | Topic | Decision |
|----|-------|----------|
| D-01 | Platform | Android only, min API 26 (Android 8.0) |
| D-02 | Model delivery | Download ONNX models from HuggingFace on first launch, cache locally |
| D-03 | Image input | Gallery pick as primary, camera capture as secondary option |
| D-04 | Output format | H.264 codec, 720p (1280×720), 17 native frames, 17 FPS (1s video) |
| D-05 | Text conditioning | Custom text prompt input with 5-6 template chips (Gentle Motion, Zoom In, Flowing, Dramatic, Cinematic Pan) |
| D-06 | UI layout | Three-screen minimalist app: Home → Generation Progress → Result/Share |
| D-07 | ONNX conversion | Include Python conversion scripts in `scripts/convert/` |

## Summary

A MAUI Android app (API 26+) that converts images to 720p @17fps videos using MobileI2V + Turbo-VAED models. Models are downloaded on first launch from HuggingFace. Users can pick/gallery or capture an image, optionally add a text prompt, and generate a 1-second H.264 video. Three-screen UI with step-by-step progress indication.