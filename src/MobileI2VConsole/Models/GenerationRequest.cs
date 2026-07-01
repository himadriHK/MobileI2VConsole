namespace MobileI2VConsole.Models;

/// <summary>
/// Input parameters for an image-to-video generation request.
/// </summary>
public record GenerationRequest
{
    /// <summary>Local path to the input image file.</summary>
    public required string ImagePath { get; init; }

    /// <summary>Optional text prompt to guide generation.</summary>
    public string? Prompt { get; init; }

    /// <summary>Output video width in pixels (default 1280 for 720p).</summary>
    public int Width { get; init; } = 1280;

    /// <summary>Output video height in pixels (default 720 for 720p).</summary>
    public int Height { get; init; } = 720;

    /// <summary>Number of frames to generate (default 17, model native).</summary>
    public int FrameCount { get; init; } = 17;

    /// <summary>Target frames per second (default 17).</summary>
    public int Fps { get; init; } = 17;

    /// <summary>Number of diffusion steps (1 or 2, default 2 for quality).</summary>
    public int DiffusionSteps { get; init; } = 2;
}
