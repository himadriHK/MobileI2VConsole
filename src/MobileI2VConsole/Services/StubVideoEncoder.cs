namespace MobileI2VConsole.Services;

/// <summary>
/// Stub encoder used for non-Android platforms (compilation only).
/// Throws <see cref="PlatformNotSupportedException"/> at runtime.
/// </summary>
public class StubVideoEncoder : IVideoEncoder
{
    public Task<string> EncodeFramesAsync(
        byte[][] frames,
        string outputPath,
        int width,
        int height,
        int fps,
        IProgress<double>? progress = null,
        CancellationToken ct = default)
    {
        throw new PlatformNotSupportedException(
            "Video encoding is only supported on Android");
    }
}
