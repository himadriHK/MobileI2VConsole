namespace MobileI2VConsole.Models;

/// <summary>
/// A predefined prompt template for quick selection.
/// </summary>
public record PromptTemplate
{
    /// <summary>Display name (e.g., "Gentle Motion").</summary>
    public required string Name { get; init; }

    /// <summary>The actual prompt text sent to the model.</summary>
    public required string Prompt { get; init; }

    /// <summary>Icon or emoji for the chip.</summary>
    public string Icon { get; init; } = "✨";
}
