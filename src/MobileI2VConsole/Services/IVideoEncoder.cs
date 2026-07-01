namespace MobileI2VConsole.Services;

/// <summary>
/// Encodes raw frame data into a video file.
/// </summary>
public interface IVideoEncoder
{
    /// <summary>
    /// Encodes raw RGBA frames into an H.264 .mp4 video file.
    /// </summary>
    /// <param name="frames">Array of raw RGBA byte arrays (one per frame).</param>
    /// <param name="outputPath">Path where the .mp4 file will be written.</param>
    /// <param name="width">Frame width in pixels.</param>
    /// <param name="height">Frame height in pixels.</param>
    /// <param name="fps">Target frames per second.</param>
    /// <param name="progress">Progress reporter (0.0 to 1.0).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>The path to the output .mp4 file.</returns>
    Task<string> EncodeFramesAsync(byte[][] frames,
                                    string outputPath,
                                    int width, int height, int fps,
                                    IProgress<double>? progress = null,
                                    CancellationToken ct = default);
}
