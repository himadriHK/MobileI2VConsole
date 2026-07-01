using Microsoft.ML.OnnxRuntime;
using MobileI2VConsole.Models;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text.Json;

namespace MobileI2VConsole.Services;

/// <summary>
/// Manages ONNX model downloading from HuggingFace, caching, loading, and unloading.
/// </summary>
public class ModelManagerService : IModelManager, IDisposable
{
    private readonly IFileService _fileService;
    private readonly HttpClient _httpClient;
    private readonly ConcurrentDictionary<string, InferenceSession> _sessions = new();
    private readonly ConcurrentDictionary<string, ModelStatus> _status = new();
    private OrtEnv? _ortEnv;
    private Dictionary<string, ModelManifestEntry>? _manifest;

    // Model registry
    private static readonly ModelInfo[] Models = new[]
    {
        new ModelInfo("vae_encoder", "https://huggingface.co/hustvl/MobileI2V/resolve/main/vae_encoder.onnx", 1),
        new ModelInfo("qwen2_encoder", "https://huggingface.co/hustvl/MobileI2V/resolve/main/qwen2_encoder.onnx", 2),
        new ModelInfo("mobilei2v_unet", "https://huggingface.co/hustvl/MobileI2V/resolve/main/mobilei2v_unet.onnx", 3),
        new ModelInfo("turbo_vaed", "https://huggingface.co/hustvl/Turbo-VAED/resolve/main/turbo_vaed.onnx", 4),
    };

    private record ModelInfo(string Name, string Url, int Order);
    private record ModelManifestEntry(string FileName, long Size, string Sha256);

    public ModelManagerService(IFileService fileService)
    {
        _fileService = fileService;
        _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };

        // Initialize status entries
        foreach (var model in Models)
        {
            _status[model.Name] = new ModelStatus
            {
                ModelName = model.Name,
                DownloadProgress = 0,
                IsDownloaded = false,
                IsLoaded = false
            };
        }
    }

    public async Task<bool> DownloadModelsAsync(IProgress<ModelStatus>? progress, CancellationToken ct)
    {
        await _fileService.InitializeAsync();
        var modelsDir = _fileService.GetModelsDirectory();

        // First download manifest
        await DownloadManifestAsync(modelsDir, ct);

        foreach (var model in Models.OrderBy(m => m.Order))
        {
            ct.ThrowIfCancellationRequested();
            var targetPath = Path.Combine(modelsDir, $"{model.Name}.onnx");

            if (File.Exists(targetPath))
            {
                var status = _status[model.Name] with { IsDownloaded = true, DownloadProgress = 1.0 };
                _status[model.Name] = status;
                progress?.Report(status);
                continue;
            }

            // Download with progress and retry
            var tempPath = targetPath + ".tmp";
            const int maxRetries = 3;
            var attempt = 0;
            var success = false;

            while (!success && attempt < maxRetries)
            {
                attempt++;
                try
                {
                    using var response = await _httpClient.GetAsync(model.Url, HttpCompletionOption.ResponseHeadersRead, ct);
                    response.EnsureSuccessStatusCode();
                    var total = response.Content.Headers.ContentLength ?? -1;
                    using var stream = await response.Content.ReadAsStreamAsync(ct);
                    using var fileStream = File.Create(tempPath);

                    var buffer = new byte[81920];
                    long bytesRead = 0;
                    int read;
                    while ((read = await stream.ReadAsync(buffer, ct)) > 0)
                    {
                        await fileStream.WriteAsync(buffer.AsMemory(0, read), ct);
                        bytesRead += read;
                        if (total > 0)
                        {
                            var pct = (double)bytesRead / total;
                            var status = _status[model.Name] with { DownloadProgress = pct };
                            _status[model.Name] = status;
                            progress?.Report(status);
                        }
                    }

                    // Verify SHA256
                    if (_manifest?.TryGetValue(model.Name, out var entry) == true && !string.IsNullOrEmpty(entry.Sha256))
                    {
                        var hash = await ComputeSha256Async(tempPath);
                        if (!string.Equals(hash, entry.Sha256, StringComparison.OrdinalIgnoreCase))
                        {
                            File.Delete(tempPath);
                            throw new InvalidDataException($"SHA256 mismatch for {model.Name}");
                        }
                    }

                    File.Move(tempPath, targetPath, overwrite: true);

                    var doneStatus = _status[model.Name] with { IsDownloaded = true, DownloadProgress = 1.0 };
                    _status[model.Name] = doneStatus;
                    progress?.Report(doneStatus);
                    success = true;
                }
                catch (OperationCanceledException)
                {
                    if (File.Exists(tempPath)) File.Delete(tempPath);
                    throw;
                }
                catch
                {
                    if (File.Exists(tempPath)) File.Delete(tempPath);
                    if (attempt >= maxRetries)
                        throw;
                    // Brief delay before retry
                    await Task.Delay(1000 * attempt, ct);
                }
            }
        }
        return true;
    }

    public Task<bool> LoadModelAsync(string modelName, CancellationToken ct)
    {
        if (_sessions.ContainsKey(modelName))
            return Task.FromResult(true);

        var modelPath = GetModelPath(modelName);
        if (!File.Exists(modelPath))
            return Task.FromResult(false);

        if (_ortEnv == null || _ortEnv.IsInvalid)
            _ortEnv = OrtEnv.Instance();
        var opts = new SessionOptions();

        // Try to enable NNAPI for hardware acceleration
        try { opts.AppendExecutionProvider_Nnapi(/* nnapiFlags */ 0); } catch { /* fall back to CPU */ }

        var session = new InferenceSession(modelPath, opts);
        _sessions[modelName] = session;

        _status[modelName] = _status[modelName] with { IsLoaded = true };
        return Task.FromResult(true);
    }

    public Task UnloadModelAsync(string modelName)
    {
        if (_sessions.TryRemove(modelName, out var session))
        {
            session.Dispose();
            _status[modelName] = _status[modelName] with { IsLoaded = false };
        }
        return Task.CompletedTask;
    }

    public bool IsModelDownloaded(string modelName) =>
        _status.TryGetValue(modelName, out var s) && s.IsDownloaded;

    public bool IsModelLoaded(string modelName) =>
        _status.TryGetValue(modelName, out var s) && s.IsLoaded;

    public string GetModelPath(string modelName) =>
        Path.Combine(_fileService.GetModelsDirectory(), $"{modelName}.onnx");

    public IReadOnlyList<ModelStatus> GetAllStatus() =>
        Models.Select(m => _status.GetValueOrDefault(m.Name, new ModelStatus
        {
            ModelName = m.Name,
            DownloadProgress = 0,
            IsDownloaded = false,
            IsLoaded = false
        })).ToList().AsReadOnly();

    public async Task ClearCacheAsync()
    {
        foreach (var model in Models)
        {
            await UnloadModelAsync(model.Name);
        }
        var dir = _fileService.GetModelsDirectory();
        if (Directory.Exists(dir))
        {
            foreach (var file in Directory.GetFiles(dir))
            {
                try { File.Delete(file); } catch { }
            }
        }
        foreach (var model in Models)
        {
            _status[model.Name] = _status[model.Name] with { IsDownloaded = false, DownloadProgress = 0, IsLoaded = false };
        }
    }

    public void Dispose()
    {
        foreach (var session in _sessions.Values)
            session.Dispose();
        _sessions.Clear();
        _httpClient.Dispose();
    }

    private static async Task<string> ComputeSha256Async(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var sha256 = SHA256.Create();
        var hash = await sha256.ComputeHashAsync(stream);
        return Convert.ToHexStringLower(hash);
    }

    private async Task DownloadManifestAsync(string modelsDir, CancellationToken ct)
    {
        var manifestUrl = "https://huggingface.co/hustvl/MobileI2V/resolve/main/models_manifest.json";

        try
        {
            var json = await _httpClient.GetStringAsync(manifestUrl, ct);
            var entries = JsonSerializer.Deserialize<Dictionary<string, ModelManifestEntry>>(json);
            _manifest = entries;
        }
        catch
        {
            // Manifest not available — proceed without SHA256 verification
            _manifest = null;
        }
    }
}
