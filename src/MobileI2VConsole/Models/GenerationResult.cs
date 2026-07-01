namespace MobileI2VConsole.Models;

/// <summary>
/// Result of a completed image-to-video generation.
/// </summary>
public record GenerationResult
{
    /// <summary>Local path to the output .mp4 video file.</summary>
    public required string VideoPath { get; init; }

    /// <summary>Duration of the generation process.</summary>
    public TimeSpan Elapsed { get; init; }

    /// <summary>Whether generation succeeded.</summary>
    public bool Success { get; init; } = true;

    /// <summary>Error message if generation failed.</summary>
    public string? ErrorMessage { get; init; }
}
