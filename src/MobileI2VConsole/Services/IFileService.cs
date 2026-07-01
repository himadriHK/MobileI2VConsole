namespace MobileI2VConsole.Services;

/// <summary>
/// Manages app file system paths and cleanup.
/// </summary>
public interface IFileService
{
    /// <summary>Directory where downloaded models are stored.</summary>
    string GetModelsDirectory();

    /// <summary>Directory where generated videos are saved.</summary>
    string GetOutputDirectory();

    /// <summary>Directory for temporary files during generation.</summary>
    string GetTempDirectory();

    /// <summary>Ensures all app directories exist, creating them if needed.</summary>
    Task InitializeAsync();

    /// <summary>Cleans up temp files older than the specified age.</summary>
    Task CleanupTempAsync(TimeSpan maxAge);

    /// <summary>Gets the file size in bytes.</summary>
    long GetFileSize(string path);
}
