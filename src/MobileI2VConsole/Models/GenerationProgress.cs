namespace MobileI2VConsole.Models;

/// <summary>
/// Progress information during video generation.
/// </summary>
public record GenerationProgress
{
    /// <summary>Name of the current pipeline step.</summary>
    public required string StepName { get; init; }

    /// <summary>Progress within current step (0.0 to 1.0).</summary>
    public double Progress { get; init; }

    /// <summary>Overall progress across all steps (0.0 to 1.0).</summary>
    public double OverallProgress { get; init; }

    /// <summary>Current frame being decoded (if applicable).</summary>
    public int CurrentFrame { get; init; }

    /// <summary>Total frames to decode (if applicable).</summary>
    public int TotalFrames { get; init; }
}
