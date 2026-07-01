using MobileI2VConsole.Models;

namespace MobileI2VConsole.Services;

/// <summary>
/// Orchestrates the full image-to-video inference pipeline.
/// </summary>
public interface IInferenceOrchestrator
{
    /// <summary>
    /// Runs the full pipeline: load models → encode image → encode text →
    /// UNet denoising (2-step) → VAE decode → return raw frames.
    /// </summary>
    /// <param name="request">Generation parameters.</param>
    /// <param name="progress">Progress reporter for UI updates.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Array of raw RGBA byte arrays, one per frame.</returns>
    Task<byte[][]> GenerateVideoAsync(GenerationRequest request,
                                       IProgress<GenerationProgress>? progress = null,
                                       CancellationToken ct = default);
}
