using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MobileI2VConsole.Models;
using MobileI2VConsole.Services;
using System.Collections.ObjectModel;
using System.Text.Json;
using Microsoft.Maui.Storage;

namespace MobileI2VConsole.ViewModels;

public partial class HomeViewModel : ObservableObject
{
    private readonly IMediaPickerService _mediaPicker;
    private readonly IFileService _fileService;
    private readonly IModelManager _modelManager;

    [ObservableProperty]
    private string? selectedImagePath;

    [ObservableProperty]
    private string? promptText;

    [ObservableProperty]
    private ImageSource? imagePreview;

    [ObservableProperty]
    private bool isModelDownloaded;

    [ObservableProperty]
    private bool isDownloading;

    [ObservableProperty]
    private double downloadProgress;

    [ObservableProperty]
    private string downloadStatusText = "Models need to be downloaded";

    public ObservableCollection<PromptTemplate> PromptTemplates { get; } = new();

    public bool IsDownloadNeeded => !IsModelDownloaded;

    partial void OnIsModelDownloadedChanged(bool value) => OnPropertyChanged(nameof(IsDownloadNeeded));

    public HomeViewModel(IMediaPickerService mediaPicker, IFileService fileService, IModelManager modelManager)
    {
        _mediaPicker = mediaPicker;
        _fileService = fileService;
        _modelManager = modelManager;
        LoadPromptTemplates();
        _ = CheckAndExtractModelsAsync();
    }

    partial void OnSelectedImagePathChanged(string? value)
    {
        if (value != null && File.Exists(value))
        {
            ImagePreview = ImageSource.FromFile(value);
        }
    }

    private void LoadPromptTemplates()
    {
        try
        {
            // Try loading from embedded resource
            var assembly = typeof(HomeViewModel).Assembly;
            using var stream = FileSystem.OpenAppPackageFileAsync("PromptTemplates.json").Result;
            if (stream != null)
            {
                using var reader = new StreamReader(stream);
                var json = reader.ReadToEnd();
                var templates = JsonSerializer.Deserialize<List<PromptTemplate>>(json);
                if (templates != null)
                {
                    PromptTemplates.Clear();
                    foreach (var t in templates)
                        PromptTemplates.Add(t);
                }
            }
        }
        catch(Exception ex)
        {
            // Fallback defaults
            PromptTemplates.Add(new PromptTemplate { Name = "Gentle Motion", Prompt = "gentle motion, smooth and subtle movement", Icon = "🌊" });
            PromptTemplates.Add(new PromptTemplate { Name = "Cinematic Pan", Prompt = "cinematic panning shot, slow camera movement", Icon = "🎬" });
        }
    }

    [RelayCommand]
    private async Task PickImageAsync()
    {
        var result = await _mediaPicker.PickImageAsync();
        if (result != null)
        {
            SelectedImagePath = result.FullPath;
        }
    }

    [RelayCommand]
    private async Task CaptureImageAsync()
    {
        var result = await _mediaPicker.CaptureImageAsync();
        if (result != null)
        {
            SelectedImagePath = result.FullPath;
        }
    }

    [RelayCommand]
    private void ApplyTemplate(PromptTemplate template)
    {
        PromptText = template.Prompt;
    }

    [RelayCommand]
    private async Task DownloadModelsAsync()
    {
        IsDownloading = true;
        DownloadStatusText = "Downloading models...";
        try
        {
            var progress = new Progress<ModelStatus>(status =>
            {
                DownloadProgress = status.DownloadProgress;
                DownloadStatusText = $"Downloading {status.ModelName}... {status.DownloadProgress:P0}";
            });

            var success = true;// all models are bundled in assets, so no download is needed. //await _modelManager.DownloadModelsAsync(progress);
            if (success)
            {
                IsModelDownloaded = true;
                DownloadStatusText = "All models ready";
            }
        }
        catch (Exception ex)
        {
            DownloadStatusText = $"Download failed: {ex.Message}";
        }
        finally
        {
            IsDownloading = false;
        }
    }

    [RelayCommand]
    private async Task GenerateAsync()
    {
        if (string.IsNullOrEmpty(SelectedImagePath))
        {
            await Shell.Current.DisplayAlert("No Image", "Please select an image first.", "OK");
            return;
        }

        if (!IsModelDownloaded)
        {
            await Shell.Current.DisplayAlert("Models Not Ready", 
                "Please download the AI models first.", "OK");
            return;
        }

        var request = new GenerationRequest
        {
            ImagePath = SelectedImagePath,
            Prompt = PromptText,
            Width = 1280,
            Height = 720,
            FrameCount = 17,
            Fps = 17,
            DiffusionSteps = 2
        };

        var serialized = JsonSerializer.Serialize(request);
        await Shell.Current.GoToAsync("generation", new Dictionary<string, object>
        {
            ["Request"] = serialized
        });
    }

    [RelayCommand]
    private async Task CheckModelStatusAsync()
    {
        await CheckAndExtractModelsAsync();
    }

    private async Task CheckAndExtractModelsAsync()
    {
        var statuses = _modelManager.GetAllStatus();
        if (statuses.All(s => s.IsDownloaded))
        {
            IsModelDownloaded = true;
            DownloadStatusText = "All models ready";
            return;
        }

        // Try extracting bundled models from app package (Resources/Raw)
        DownloadStatusText = "Extracting bundled models...";
        try
        {
            var success = await _modelManager.DownloadModelsAsync(null);
            if (success)
            {
                IsModelDownloaded = true;
                DownloadStatusText = "All models ready";
            }
            else
            {
                DownloadStatusText = "Models need to be downloaded";
            }
        }
        catch
        {
            DownloadStatusText = "Models need to be downloaded";
        }
    }
}
