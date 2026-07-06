using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MobileI2VConsole.Models;
using SkiaSharp;
using System.Threading.Tasks;

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
    private const int UnetLatentChannels = 128;
    private const int OutputFrames = 17;
    private const int VaeOutputW = 1280;
    private const int VaeOutputH = 720;

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

            var modelsToLoad = new[] { "vae_encoder.onnx", "qwen2_encoder.onnx", "mobilei2v_unet.onnx", "turbo_vaed.onnx" };
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
            // Step 5 — Turbo-VAED decode (all 17 frames in a single call)
            // ═══════════════════════════════════════════════════════════════
            ReportProgress(progress, "Decoding video", ProgressDecodingStart, 0, OutputFrames);
            ct.ThrowIfCancellationRequested();

            var frames = await RunTurboVaedDecodeAsync(denoisedLatents, progress, ct);
            ReportProgress(progress, "Decoding video", ProgressDecodingEnd, OutputFrames, OutputFrames);

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
        catch (OnnxRuntimeException ex)
        {
            Console.Error.WriteLine($"Model file corrupted during inference: {ex.Message}. Clear your app cache and re-download models.");
            throw;
        }
        catch (Exception ex)
        {
            // Log the exception (you can replace this with your preferred logging mechanism)
            Console.Error.WriteLine($"Error during video generation: {ex.Message}");
            throw;
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
        float scale = Math.Max((float)targetSize / bitmap.Width, (float)targetSize / bitmap.Height);
        int resizedW = (int)(bitmap.Width * scale);
        int resizedH = (int)(bitmap.Height * scale);

        // Resize to a known pixel format (Rgba8888) for reliable byte-level access
        using var resized = bitmap.Resize(
            new SKImageInfo(resizedW, resizedH, SKColorType.Rgba8888, SKAlphaType.Unpremul),
            new SKSamplingOptions(SKFilterMode.Linear));
        if (resized == null)
            throw new InvalidOperationException("Failed to resize image.");

        // Get raw pixel bytes — avoids per-pixel P/Invoke marshaling issues
        int bpp = resized.BytesPerPixel;
        int rowBytes = resized.RowBytes;
        int totalBytes = rowBytes * resized.Height;
        IntPtr pixelsPtr = resized.GetPixels();
        byte[] pixelBytes = new byte[totalBytes];
        System.Runtime.InteropServices.Marshal.Copy(pixelsPtr, pixelBytes, 0, totalBytes);

        // Centre-crop and normalise [0, 255] → [-1, 1] in CHW layout in one pass
        int cropX = Math.Max(0, (resized.Width - targetSize) / 2);
        int cropY = Math.Max(0, (resized.Height - targetSize) / 2);
        var result = new float[1 * 3 * targetSize * targetSize];
        int area = targetSize * targetSize;
        int idx = 0;

        for (int y = 0; y < targetSize; y++)
        {
            int rowStart = (cropY + y) * rowBytes;
            for (int x = 0; x < targetSize; x++)
            {
                int offset = rowStart + (cropX + x) * bpp;
                byte r = pixelBytes[offset];
                byte g = pixelBytes[offset + 1];
                byte b = pixelBytes[offset + 2];

                result[idx]            = (r / 255f) * 2f - 1f;
                result[idx +     area] = (g / 255f) * 2f - 1f;
                result[idx + 2 * area] = (b / 255f) * 2f - 1f;
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
            if (!_modelManager.TryGetSession("vae_encoder.onnx", out var session))
                throw new InvalidOperationException("VAE encoder model not loaded.");
            // Diagnostic: log input metadata to verify expected element types
            try
            {
                Console.Error.WriteLine("VAE session input metadata:");
                foreach (var kv in session.InputMetadata)
                    Console.Error.WriteLine($"  {kv.Key}: {kv.Value.ElementType} dims=[{string.Join(',', kv.Value.Dimensions)}]");
            }
            catch { /* best-effort logging */ }

            var inputs = new[] { CreateNamedValueForFloatData(session, "pixel_values", preprocessedImage, new[] { 1, 3, VaeInputSize, VaeInputSize }) };
            using var results = session.Run(inputs, new[] { "latent" });
            return ReadFloat16Output(results[0]);
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
                return new float[1 * 1 * 896];
            }

            if (!_modelManager.TryGetSession("qwen2_encoder.onnx", out var session))
                throw new InvalidOperationException("Qwen2 encoder model not loaded.");

            const int maxSeqLen = 77;
            var textBytes = System.Text.Encoding.UTF8.GetBytes(prompt);
            var tokens = new long[maxSeqLen];
            Array.Fill(tokens, 0L); // pad token
            for (int i = 0; i < Math.Min(textBytes.Length, maxSeqLen - 2); i++)
                tokens[i + 1] = textBytes[i];
            tokens[0] = 1; // <|begin_of_text|>
            tokens[Math.Min(textBytes.Length + 1, maxSeqLen - 1)] = 2; // <|end_of_text|>

            var mask = new long[maxSeqLen];
            Array.Fill(mask, 1L);

            var inputs = new[]
            {
                CreateInt64NamedValue("input_ids", tokens, new[] { 1, maxSeqLen }),
                CreateInt64NamedValue("attention_mask", mask, new[] { 1, maxSeqLen })
            };

            using var results = session.Run(inputs, new[] { "last_hidden_state" });
            return ReadFloat16Output(results[0]);
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
            const int numSteps = 2;
            int actualSteps = Math.Min(steps, numSteps);

            if (!_modelManager.TryGetSession("mobilei2v_unet.onnx", out var session))
                throw new InvalidOperationException("UNet model not loaded.");

            int singleFrameSize = LatentH * LatentW; // 64*64 = 4096
            int vaeChannels = latent.Length / singleFrameSize;

            // Tile VAE channels to 128, repeat for 17 frames
            var tiledFrame = new float[UnetLatentChannels * LatentH * LatentW];
            for (int c = 0; c < UnetLatentChannels; c++)
            {
                int srcChan = c % vaeChannels;
                Array.Copy(latent, srcChan * singleFrameSize, tiledFrame, c * singleFrameSize, singleFrameSize);
            }

            int frameLatentSize = UnetLatentChannels * LatentH * LatentW;
            var unetInput = new float[OutputFrames * frameLatentSize];
            for (int f = 0; f < OutputFrames; f++)
            {
                Array.Copy(tiledFrame, 0, unetInput, f * frameLatentSize, frameLatentSize);
            }

            // Distilled timesteps for MobileI2V 2-step
            int[] timesteps = { 999, 499 };
            float[] alphas = { 0.0001f, 0.5f };
            float[] sigmas = { 0.9999f, 0.5f };

            var currentLatent = (float[])unetInput.Clone();

            for (int step = 0; step < actualSteps; step++)
            {
                ct.ThrowIfCancellationRequested();

                double stepFraction = (double)(step + 1) / actualSteps;
                double overall = ProgressDiffusionStart +
                                 stepFraction * (ProgressDiffusionEnd - ProgressDiffusionStart);

                ReportProgress(progress, $"Denoising ({step + 1}/{actualSteps})",
                    overall, step, actualSteps);

                // Diagnostic: log UNet session input metadata
                try
                {
                    Console.Error.WriteLine($"UNet session input metadata (step {step}):");
                    foreach (var kv in session.InputMetadata)
                        Console.Error.WriteLine($"  {kv.Key}: {kv.Value.ElementType} dims=[{string.Join(',', kv.Value.Dimensions)}]");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Failed to read UNet input metadata: {ex.GetType().Name}: {ex.Message}");
                }

                var inputs = new[]
                {
                    CreateNamedValueForFloatData(session, "latent", currentLatent,
                        new[] { 1, UnetLatentChannels, OutputFrames, LatentH, LatentW }),
                    CreateInt64NamedValue("timestep", new long[] { timesteps[step] }, new[] { 1 })
                };

                // Diagnostic logging: input tensor types and shapes
                Console.Error.WriteLine($"UNet Run {step + 1} inputs:");
                foreach (var input in inputs)
                {
                Console.Error.WriteLine($"  {input.Name}: {input.ValueType}");
                }

                using var results = session.Run(inputs, new[] { "denoised_latent" });
                var noisePred = ReadFloat16Output(results[0]);

                // Apply DDIM denoising step
                int latentSize = currentLatent.Length;
                for (int i = 0; i < latentSize; i++)
                {
                    currentLatent[i] = (currentLatent[i] - sigmas[step] * noisePred[i]) / alphas[step];
                }
            }

            return currentLatent;
        }, ct);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Turbo-VAED Decoder — latent → 17 RGB frames (720p each)
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes all 17 frame latents through the Turbo-VAED decoder in a single call.
    /// Input: [1, 128, 17, 64, 64] → Output: [1, 3, 17, 720, 1280].
    /// </summary>
    /// <returns>Array of 17 float arrays, each [3, 720, 1280] in CHW layout.</returns>
    private async Task<float[][]> RunTurboVaedDecodeAsync(
        float[] latentFrames,
        IProgress<GenerationProgress>? progress, CancellationToken ct)
    {
        return await Task.Run(() =>
        {
            if (!_modelManager.TryGetSession("turbo_vaed.onnx", out var session))
                throw new InvalidOperationException("Turbo-VAED model not loaded.");

            ct.ThrowIfCancellationRequested();

            ReportProgress(progress, "Decoding video", ProgressDecodingStart, 0, OutputFrames);

            var inputs = new[] { CreateNamedValueForFloatData(session, "latent", latentFrames,
                new[] { 1, UnetLatentChannels, OutputFrames, LatentH, LatentW }) };
            // Diagnostic: log Turbo-VAED session input metadata
            try
            {
                Console.Error.WriteLine("Turbo-VAED session input metadata:");
                foreach (var kv in session.InputMetadata)
                    Console.Error.WriteLine($"  {kv.Key}: {kv.Value.ElementType} dims=[{string.Join(',', kv.Value.Dimensions)}]");
            }
            catch { }


            // Diagnostic logging: input tensor types and shapes
            Console.Error.WriteLine($"Turbo-VAED decode inputs:");
            foreach (var input in inputs)
            {
                Console.Error.WriteLine($"  {input.Name}: {input.ValueType}");
            }

            using var results = session.Run(inputs, new[] { "frames" });

            var outputArray = ReadFloat16Output(results[0]);

            var frames = new float[OutputFrames][];
            int frameSize = 3 * VaeOutputW * VaeOutputH;
            for (int i = 0; i < OutputFrames; i++)
            {
                frames[i] = new float[frameSize];
                Array.Copy(outputArray, i * frameSize, frames[i], 0, frameSize);
            }

            ReportProgress(progress, "Decoding video", ProgressDecodingEnd, OutputFrames, OutputFrames);

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

    // C#
    /// <summary>
    /// Creates a NamedOnnxValue for float data, choosing between float32 and float16
    /// depending on the session input metadata. If Half creation fails, falls back
    /// to float32 to preserve robustness on platforms with broken Half support.
    /// </summary>
    private static NamedOnnxValue CreateNamedValueForFloatData(InferenceSession session, string name, float[] data, int[] shape)
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data), $"CreateNamedValueForFloatData: '{name}' received null data.");
        long expected = 1;
        foreach (var dim in shape) expected *= dim;
        if (expected != data.Length)
            throw new ArgumentException($"CreateNamedValueForFloatData: '{name}' length mismatch: expected {expected}, got {data.Length}.");

        try
        {
            // NodeMetadata.ElementType is a TensorElementType enum. Check for
            // Float16 to decide whether to create a Half (float16) tensor.
            if (session.InputMetadata != null)
            {
                if (session.InputMetadata.TryGetValue(name, out var meta))
                {
                    Console.Error.WriteLine($"CreateNamedValueForFloatData: session has metadata for '{name}': ElementType={meta.ElementType}, dims=[{string.Join(',', meta.Dimensions)}]");
                    if (meta.ElementDataType == TensorElementType.Float16)
                    {
                        // Try to create a float16 tensor. This can throw on some runtimes; we
                        // catch and fall back to float32 below.
                        try
                        {
                            var halfData = new Half[data.Length];
                            for (int i = 0; i < data.Length; i++)
                                halfData[i] = (Half)data[i];
                            Console.Error.WriteLine($"CreateNamedValueForFloatData: '{name}' created Half tensor ({data.Length} elements).");
                            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Half>(halfData, shape));
                        }
                        catch (Exception e)
                        {
                            Console.Error.WriteLine($"CreateNamedValueForFloatData: failed to create Half tensor for '{name}': {e.Message}. Falling back to Float32.");
                            // fall through to float32 creation
                        }
                    }
                }
                else
                {
                    // Log available input names/types to help diagnose name mismatches
                    try
                    {
                        Console.Error.WriteLine($"CreateNamedValueForFloatData: session metadata does NOT contain '{name}'. Available inputs:");
                        foreach (var kv in session.InputMetadata)
                            Console.Error.WriteLine($"  {kv.Key}: {kv.Value.ElementType} dims=[{string.Join(',', kv.Value.Dimensions)}]");
                    }
                    catch { }
                }
            }
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"CreateNamedValueForFloatData: exception while inspecting session metadata: {e.Message}");
            // fall through to create float32 tensor
        }

        // Default to float32 tensor
        Console.Error.WriteLine($"CreateNamedValueForFloatData: '{name}' created Float32 tensor ({data.Length} elements).");
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, shape));
    }

    private static NamedOnnxValue CreateInt64NamedValue(string name, long[] data, int[] shape)
    {
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, shape));
    }

    private static NamedOnnxValue CreateInt32NamedValue(string name, int[] data, int[] shape)
    {
        return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(data, shape));
    }
    private static float[] ReadFloat16Output(NamedOnnxValue value)
    {
        // OnnxRuntime 1.27+: NamedOnnxValue no longer exposes .Tensor or .ElementType directly.
        // Use AsTensor<T>() with try-catch to determine the element type at runtime.
        try
        {
            var halfTensor = value.AsTensor<Half>();
            var result = new float[halfTensor.Length];
            for (int i = 0; i < halfTensor.Length; i++)
                result[i] = (float)halfTensor.GetValue(i);
            return result;
        }
        catch
        {
            // All other formats (Float32, Float64, etc.) - read as float
            var floatTensor = value.AsTensor<float>();
            var floatResult = new float[floatTensor.Length];
            for (int i = 0; i < floatTensor.Length; i++)
                floatResult[i] = floatTensor.GetValue(i);
            return floatResult;
        }
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
