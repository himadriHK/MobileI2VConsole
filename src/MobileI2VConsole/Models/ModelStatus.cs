namespace MobileI2VConsole.Models;

/// <summary>
/// Download and load status for an ONNX model.
/// </summary>
public record ModelStatus
{
    /// <summary>Name of the model (e.g., "vae_encoder", "mobilei2v_unet").</summary>
    public required string ModelName { get; init; }

    /// <summary>Current download progress (0.0 to 1.0).</summary>
    public double DownloadProgress { get; init; }

    /// <summary>Whether the model file is downloaded and verified.</summary>
    public bool IsDownloaded { get; init; }

    /// <summary>Whether the model is loaded into an ONNX session.</summary>
    public bool IsLoaded { get; init; }

    /// <summary>Error message if download/load failed.</summary>
    public string? ErrorMessage { get; init; }
}
