using MobileI2VConsole.Models;

namespace MobileI2VConsole.Services;

/// <summary>
/// Manages ONNX model downloading, caching, loading, and unloading.
/// </summary>
public interface IModelManager
{
    /// <summary>Downloads all required models from HuggingFace, with progress.</summary>
    Task<bool> DownloadModelsAsync(IProgress<ModelStatus>? progress = null,
                                   CancellationToken ct = default);

    /// <summary>Loads a specific ONNX model into memory for inference.</summary>
    Task<bool> LoadModelAsync(string modelName, CancellationToken ct = default);

    /// <summary>Unloads a model to free memory.</summary>
    Task UnloadModelAsync(string modelName);

    /// <summary>Checks if a model is downloaded and verified.</summary>
    bool IsModelDownloaded(string modelName);

    /// <summary>Checks if a model is currently loaded in memory.</summary>
    bool IsModelLoaded(string modelName);

    /// <summary>Gets the file path to a downloaded model.</summary>
    string GetModelPath(string modelName);

    /// <summary>Returns status for all models.</summary>
    IReadOnlyList<ModelStatus> GetAllStatus();

    /// <summary>Deletes all cached model files.</summary>
    Task ClearCacheAsync();
}
