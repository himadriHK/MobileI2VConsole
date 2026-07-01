using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MobileI2VConsole.Models;
using SkiaSharp;

namespace MobileI2VConsole.Services;

/// <summary>
/// Core pipeline coordinator for image-to-video generation.
/// Orchestrates VAE encode → Qwen2 text encode → UNet denoising → Turbo-VAED decode.
/// </summary>
public class InferenceOrchestrator : IInferenceOrchestrator
{
    private readonly IModelManager _modelManager;
    private readonly IFileService _fileService;

    // ── Model input dimensions ─────────────────────────────────────────────
    private const int VaeInputSize = 512;        // VAE encoder expects 512×512
    private const int LatentH = VaeInputSize / 8; // 64
    private const int LatentW = VaeInputSize / 8; // 64
    private const int UnetLatentChannels = 4;
    private const int OutputFrames = 17;

    // ── Pipeline progress boundaries ───────────────────────────────────────
    private const double ProgressLoadingModels = 0.00;
    private const double ProgressEncodingImage = 0.10;
    private const double ProgressEncodingText  = 0.15;
    private const double ProgressDiffusionStart = 0.15;
    private const double ProgressDiffusionEnd   = 0.85;
    private const double ProgressDecodingStart  = 0.85;
    private const double ProgressDecodingEnd    = 0.95;
    private const double ProgressSavingStart    = 0.95;
    private const double ProgressComplete       = 1.00;
    private const int TotalPipelineSteps        = 6;

    public InferenceOrchestrator(IModelManager modelManager, IFileService fileService)
    {
        _modelManager = modelManager;
        _fileService = fileService;
    }

    /// <inheritdoc />
    public async Task<byte[][]> GenerateVideoAsync(
        GenerationRequest request,
        IProgress<GenerationProgress>? progress = null,
        CancellationToken ct = default)
    {
        var result = new byte[OutputFrames][];

        try
        {
            // ═══════════════════════════════════════════════════════════════
            // Step 0 — Load all required models
            // ═══════════════════════════════════════════════════════════════
            ReportProgress(progress, "Loading models", ProgressLoadingModels, 0, 0);
            ct.ThrowIfCancellationRequested();

            var modelsToLoad = new[] { "vae_encoder", "qwen2_encoder", "mobilei2v_unet", "turbo_vaed" };
            foreach (var model in modelsToLoad)
            {
                ct.ThrowIfCancellationRequested();

                if (_modelManager.IsModelLoaded(model))
                    continue;

                var loaded = await _modelManager.LoadModelAsync(model, ct);
                if (!loaded)
                    throw new InvalidOperationException($"Failed to load model: {model}");
            }

            // ═══════════════════════════════════════════════════════════════
            // Step 1 — Load image, preprocess, run VAE encoder
            // ═══════════════════════════════════════════════════════════════
            ReportProgress(progress, "Encoding image", ProgressEncodingImage, 0, 0);
            ct.ThrowIfCancellationRequested();

            var inputTensor = LoadAndPreprocessImage(request.ImagePath, VaeInputSize);
            var latent = await RunVaeEncoderAsync(inputTensor, ct);

            // ═══════════════════════════════════════════════════════════════
            // Step 2 — Encode text prompt via Qwen2 encoder
            // ═══════════════════════════════════════════════════════════════
            ReportProgress(progress, "Encoding prompt", ProgressEncodingText, 0, 0);
            ct.ThrowIfCancellationRequested();

            var textEmbeddings = await EncodeTextPromptAsync(request.Prompt ?? "", ct);

            // ═══════════════════════════════════════════════════════════════
            // Steps 3 & 4 — UNet 2-step denoising diffusion
            // ═══════════════════════════════════════════════════════════════
            var denoisedLatents = await RunUnetDenoisingAsync(
                latent, textEmbeddings, request.DiffusionSteps, progress, ct);

            // ═══════════════════════════════════════════════════════════════
            // Step 5 — Turbo-VAED decode (17 frames)
            // ═══════════════════════════════════════════════════════════════
            ReportProgress(progress, "Decoding video", ProgressDecodingStart, 0, OutputFrames);
            ct.ThrowIfCancellationRequested();

            var frames = await RunTurboVaedDecodeAsync(denoisedLatents, progress, ct);

            // ═══════════════════════════════════════════════════════════════
            // Step 6 — Post-process: float32 RGB → byte RGBA
            // ═══════════════════════════════════════════════════════════════
            ReportProgress(progress, "Saving output", ProgressSavingStart, OutputFrames, OutputFrames);

            for (int i = 0; i < frames.Length; i++)
            {
                result[i] = ConvertFloatsToRgba(frames[i], request.Width, request.Height);
            }

            ReportProgress(progress, "Complete", ProgressComplete, OutputFrames, OutputFrames);

            return result;
        }
        finally
        {
            // Free model memory — unload in reverse dependency order
            foreach (var model in new[] { "turbo_vaed", "mobilei2v_unet", "qwen2_encoder", "vae_encoder" })
            {
                try { await _modelManager.UnloadModelAsync(model); }
                catch { /* best-effort cleanup */ }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Image pre-processing
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Loads an image from disk, resizes to <paramref name="targetSize"/> while
    /// maintaining aspect ratio, centre-crops, and normalises pixels to [-1, 1].
    /// </summary>
    /// <returns>Float array in CHW layout: [1, 3, targetSize, targetSize].</returns>
    private static float[] LoadAndPreprocessImage(string imagePath, int targetSize)
    {
        using var input = File.OpenRead(imagePath);
        using var bitmap = SKBitmap.Decode(input);

        if (bitmap == null)
            throw new InvalidDataException("Failed to decode input image.");

        // Resize — fit within targetSize while preserving aspect ratio
        float scale = Math.Min((float)targetSize / bitmap.Width, (float)targetSize / bitmap.Height);
        int resizedW = (int)(bitmap.Width * scale);
        int resizedH = (int)(bitmap.Height * scale);

        using var resized = bitmap.Resize(new SKImageInfo(resizedW, resizedH), new SKSamplingOptions(SKFilterMode.Linear));
        if (resized == null)
            throw new InvalidOperationException("Failed to resize image.");

        // Centre-crop to exactly targetSize × targetSize
        int cropX = (resized.Width - targetSize) / 2;
        int cropY = (resized.Height - targetSize) / 2;
        using var cropped = new SKBitmap(targetSize, targetSize);
        if (!resized.ExtractSubset(cropped, new SKRectI(cropX, cropY, cropX + targetSize, cropY + targetSize)))
            throw new InvalidOperationException("Failed to centre-crop image.");

        // Normalise [0, 255] → [-1, 1] in CHW layout
        var result = new float[1 * 3 * targetSize * targetSize];
        int idx = 0;
        for (int y = 0; y < targetSize; y++)
        {
            for (int x = 0; x < targetSize; x++)
            {
                var pixel = cropped.GetPixel(x, y);
                result[idx]                              = (pixel.Red   / 255f) * 2f - 1f; // R
                result[idx +       targetSize * targetSize] = (pixel.Green / 255f) * 2f - 1f; // G
                result[idx + 2 * targetSize * targetSize] = (pixel.Blue  / 255f) * 2f - 1f; // B
                idx++;
            }
        }

        return result;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  VAE Encoder — image pixels → latent
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the VAE encoder ONNX model: [1, 3, 512, 512] → [1, 4, 64, 64].
    /// </summary>
    private async Task<float[]> RunVaeEncoderAsync(float[] preprocessedImage, CancellationToken ct)
    {
        return await Task.Run(() =>
        {
            // TODO: Implement actual ONNX inference — placeholder for now.
            //
            // Actual implementation:
            //   1. Get session via model manager (or create from model path)
            //   2. Create OrtValue tensor:
            //      using var inputOrt = OrtValue.CreateTensorValueFromMemory(
            //          OrtAllocator.DefaultInstance,
            //          preprocessedImage,
            //          new long[] { 1, 3, VaeInputSize, VaeInputSize });
            //   3. session.Run() with input / output names
            //   4. Extract output tensor data as float[]
            //
            // Return latent shape: [1, 4, 64, 64]
            return new float[1 * UnetLatentChannels * LatentH * LatentW];
        }, ct);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Qwen2 Text Encoder — prompt string → text embeddings
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Tokenises the prompt and runs the Qwen2 encoder ONNX model.
    /// Returns embeddings: [1, sequence_len, 1024].
    /// For an empty prompt, returns zeroed [1, 1, 1024].
    /// </summary>
    private async Task<float[]> EncodeTextPromptAsync(string prompt, CancellationToken ct)
    {
        return await Task.Run(() =>
        {
            if (string.IsNullOrWhiteSpace(prompt))
            {
                // Empty prompt — return minimal placeholder embeddings
                return new float[1 * 1 * 1024];
            }

            // TODO: Implement actual tokenization + ONNX inference — placeholder for now.
            //
            // Actual implementation:
            //   1. Tokenize prompt using Qwen2 tokenizer (byte-level BPE)
            //   2. Pad/truncate to sequence_len (e.g. 77 or 16 tokens)
            //   3. Create int64 input_ids tensor
            //   4. Run Qwen2 encoder session
            //   5. Extract last hidden state [1, seq_len, 1024]
            //
            // Return dummy: [1, 16, 1024] — sequence_len=16, hidden=1024
            return new float[1 * 16 * 1024];
        }, ct);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  UNet 2-Step Denoising — latent + text → denoised 17-frame latent
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the MobileI2V UNet for the specified number of diffusion steps.
    /// Each step predicts noise residuals and denoises the latent.
    /// After denoising, expands the single latent into 17-frame output.
    /// </summary>
    private async Task<float[]> RunUnetDenoisingAsync(
        float[] latent, float[] textEmbeddings, int steps,
        IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        return await Task.Run(() =>
        {
            const int numSteps = 2; // distilled turbo model
            int actualSteps = Math.Min(steps, numSteps);

            for (int step = 0; step < actualSteps; step++)
            {
                ct.ThrowIfCancellationRequested();

                double stepFraction = (double)(step + 1) / actualSteps;
                double overall = ProgressDiffusionStart +
                                 stepFraction * (ProgressDiffusionEnd - ProgressDiffusionStart);

                ReportProgress(progress, $"Generating frames ({step + 1}/{actualSteps})",
                    overall, step, actualSteps);

                // TODO: Implement actual ONNX inference — placeholder for now.
                //
                // Actual implementation:
                //   1. Compute timestep embedding for current step
                //   2. Expand latent for temporal sliding window (9 frames)
                //   3. Create input tensors: noisy_latent + text_embeddings + timestep
                //   4. session.Run() on "mobilei2v_unet"
                //   5. Output is denoised latent [1, 4, 64, 64]
            }

            // After denoising, expand single latent to 17 frames
            // TODO: Apply temporal shifts per frame to produce 17-frame output
            var result = new float[OutputFrames * UnetLatentChannels * LatentH * LatentW];
            return result;
        }, ct);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Turbo-VAED Decoder — latent → 17 RGB frames (720p each)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes each of the 17 frame latents through the Turbo-VAED decoder.
    /// Each frame is decoded independently.
    /// </summary>
    /// <returns>Array of 17 float arrays, each [3, 720, 1280] in CHW layout.</returns>
    private async Task<float[][]> RunTurboVaedDecodeAsync(
        float[] latentFrames,
        IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        var frames = new float[OutputFrames][];

        return await Task.Run(() =>
        {
            // TODO: Implement actual ONNX inference — placeholder for now.
            //
            // Actual implementation:
            //   1. Get turbo_vaed session
            //   2. For each frame i:
            //      a. Extract per-frame latent [1, 4, 64, 64]
            //      b. Create OrtValue tensor
            //      c. session.Run() — output shape [1, 3, 720, 1280]
            //      d. Copy output data to frames[i]

            // Placeholder: 720p RGB float data
            int frameSize = 3 * 1280 * 720;

            for (int i = 0; i < OutputFrames; i++)
            {
                ct.ThrowIfCancellationRequested();
                frames[i] = new float[frameSize];

                ReportProgress(progress, "Decoding video",
                    ProgressDecodingStart + (i + 1) * (ProgressDecodingEnd - ProgressDecodingStart) / OutputFrames,
                    i + 1, OutputFrames);
            }

            return frames;
        }, ct);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Colour-space conversion — float32 RGB → byte RGBA
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Converts a float32 RGB frame (CHW layout, values in [-1, 1]) to a
    /// byte RGBA array suitable for video encoding.
    /// </summary>
    private static byte[] ConvertFloatsToRgba(float[] rgbFrame, int width, int height)
    {
        int pixelCount = width * height;
        var result = new byte[pixelCount * 4];

        for (int i = 0; i < pixelCount; i++)
        {
            // Denormalise: [-1, 1] → [0, 255]
            byte r = (byte)Math.Clamp((rgbFrame[i] + 1f) * 127.5f, 0, 255);
            byte g = (byte)Math.Clamp((rgbFrame[i + pixelCount] + 1f) * 127.5f, 0, 255);
            byte b = (byte)Math.Clamp((rgbFrame[i + 2 * pixelCount] + 1f) * 127.5f, 0, 255);

            result[i * 4]     = r;
            result[i * 4 + 1] = g;
            result[i * 4 + 2] = b;
            result[i * 4 + 3] = 255; // fully opaque alpha
        }

        return result;
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Helpers
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Retrieves an ONNX runtime session for the named model.
    /// </summary>
    /// <remarks>
    /// TODO: This will eventually retrieve the session from <see cref="IModelManager"/>.
    /// The current implementation is a placeholder since <c>IModelManager</c> does not
    /// yet expose a <c>GetSession</c> method. The pipeline methods currently return
    /// dummy data instead of running actual ONNX inference.
    ///
    /// Expected future implementation:
    /// <code>
    /// var path = _modelManager.GetModelPath(modelName);
    /// var env = OrtEnvironment.GetEnvironment();
    /// var opts = new OrtSession.SessionOptions();
    /// opts.AppendExecutionProvider_Nnapi(0);
    /// return new OrtSession(env, path, opts);
    /// </code>
    /// </remarks>
    [Obsolete("ONNX inference not yet wired — placeholder session accessor.")]
    private static InferenceSession GetSession(string modelName)
    {
        throw new NotImplementedException(
            $"ONNX inference requires an actual OrtSession for '{modelName}'. " +
            "Retrieve it from IModelManager once GetSession() is exposed, " +
            "or create one via OrtEnvironment + model path.");
    }

    /// <summary>
    /// Reports pipeline progress through <see cref="IProgress{T}"/>.
    /// </summary>
    private static void ReportProgress(
        IProgress<GenerationProgress>? progress,
        string stepName,
        double overallProgress,
        int currentFrame,
        int totalFrames)
    {
        progress?.Report(new GenerationProgress
        {
            StepName = stepName,
            Progress = totalFrames > 0 ? (double)currentFrame / totalFrames : 0.0,
            OverallProgress = Math.Clamp(overallProgress, 0.0, 1.0),
            CurrentFrame = currentFrame,
            TotalFrames = totalFrames
        });
    }
}
