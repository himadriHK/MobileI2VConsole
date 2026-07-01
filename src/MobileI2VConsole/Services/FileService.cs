namespace MobileI2VConsole.Services;

/// <summary>
/// Manages app file system paths and cleanup.
/// </summary>
public class FileService : IFileService
{
    private readonly string _modelsDir;
    private readonly string _outputDir;
    private readonly string _tempDir;

    public FileService()
    {
        var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.Personal);
        _modelsDir = Path.Combine(baseDir, "models");
        _outputDir = Path.Combine(baseDir, "output");
        _tempDir = Path.Combine(baseDir, "temp");
    }

    public string GetModelsDirectory() => _modelsDir;
    public string GetOutputDirectory() => _outputDir;
    public string GetTempDirectory() => _tempDir;

    public async Task InitializeAsync()
    {
        foreach (var dir in new[] { _modelsDir, _outputDir, _tempDir })
        {
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);
        }
        await Task.CompletedTask;
    }

    public async Task CleanupTempAsync(TimeSpan maxAge)
    {
        if (!Directory.Exists(_tempDir)) return;
        var files = Directory.GetFiles(_tempDir);
        foreach (var file in files)
        {
            var info = new FileInfo(file);
            if (DateTime.UtcNow - info.CreationTimeUtc > maxAge)
            {
                try { File.Delete(file); } catch { /* best effort */ }
            }
        }
        await Task.CompletedTask;
    }

    public long GetFileSize(string path) =>
        File.Exists(path) ? new FileInfo(path).Length : 0;
}
