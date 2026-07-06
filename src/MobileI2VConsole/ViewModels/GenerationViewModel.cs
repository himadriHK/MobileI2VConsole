using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MobileI2VConsole.Models;
using MobileI2VConsole.Services;
using System.Text.Json;

namespace MobileI2VConsole.ViewModels;

[QueryProperty(nameof(Request), "Request")]
public partial class GenerationViewModel : ObservableObject
{
    private readonly IInferenceOrchestrator _orchestrator;
    private readonly IVideoEncoder _videoEncoder;
    private readonly IFileService _fileService;
    private CancellationTokenSource? _cts;

    [ObservableProperty]
    private string? request;

    [ObservableProperty]
    private string stepName = "Preparing...";

    [ObservableProperty]
    private double progress;

    [ObservableProperty]
    private string progressText = "";

    [ObservableProperty]
    private bool isRunning = true;

    [ObservableProperty]
    private bool isCompleted;

    [ObservableProperty]
    private string? resultPath;

    [ObservableProperty]
    private string? errorMessage;

    public GenerationViewModel(
        IInferenceOrchestrator orchestrator,
        IVideoEncoder videoEncoder,
        IFileService fileService)
    {
        _orchestrator = orchestrator;
        _videoEncoder = videoEncoder;
        _fileService = fileService;
    }

    partial void OnRequestChanged(string? value)
    {
        if (value != null)
        {
            // Start generation automatically when request is received
            _ = StartGenerationAsync();
        }
    }

    private async Task StartGenerationAsync()
    {
        if (string.IsNullOrEmpty(Request)) return;

        _cts = new CancellationTokenSource();
        IsRunning = true;
        IsCompleted = false;
        ErrorMessage = null;

        try
        {
            var genRequest = JsonSerializer.Deserialize<GenerationRequest>(Request);
            if (genRequest == null)
            {
                ErrorMessage = "Invalid generation request";
                return;
            }

            var progress = new Progress<GenerationProgress>(p =>
            {
                StepName = p.StepName;
                Progress = p.OverallProgress;
                if (p.TotalFrames > 0)
                    ProgressText = $"Frame {p.CurrentFrame}/{p.TotalFrames}";
                else
                    ProgressText = $"{p.OverallProgress:P0}";
            });

            // Run inference pipeline (background thread to keep UI responsive)
            var rawFrames = await Task.Run(
                () => _orchestrator.GenerateVideoAsync(genRequest, progress, _cts.Token), _cts.Token);

            // Encode video
            StepName = "Encoding video";
            var outputPath = Path.Combine(_fileService.GetOutputDirectory(),
                $"video_{DateTime.Now:yyyyMMddHHmmss}.mp4");

            var encodeProgress = new Progress<double>(p =>
            {
                Progress = 0.9 + p * 0.1;
                ProgressText = $"Encoding {p:P0}";
            });

            var videoPath = await _videoEncoder.EncodeFramesAsync(
                rawFrames, outputPath, genRequest.Width, genRequest.Height,
                genRequest.Fps, encodeProgress, _cts.Token);

            ResultPath = videoPath;
            IsCompleted = true;
        }
        catch (OperationCanceledException)
        {
            ErrorMessage = "Generation was cancelled";
        }
        catch (Exception ex)
        {
            ErrorMessage = $"Generation failed: {ex.Message}";
        }
        finally
        {
            IsRunning = false;
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        _cts?.Cancel();
        StepName = "Cancelling...";
    }

    [RelayCommand]
    private async Task ViewResultAsync()
    {
        if (!string.IsNullOrEmpty(ResultPath))
        {
            await Shell.Current.GoToAsync("result", new Dictionary<string, object>
            {
                ["VideoPath"] = ResultPath
            });
        }
    }
}
