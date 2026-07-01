using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MobileI2VConsole.Models;

namespace MobileI2VConsole.ViewModels;

[QueryProperty(nameof(VideoPath), "VideoPath")]
public partial class ResultViewModel : ObservableObject
{
    [ObservableProperty]
    private string? videoPath;

    [ObservableProperty]
    private ImageSource? videoSource;

    public ResultViewModel()
    {
    }

    partial void OnVideoPathChanged(string? value)
    {
        if (!string.IsNullOrEmpty(value) && File.Exists(value))
        {
            VideoSource = ImageSource.FromFile(value);
        }
    }

    [RelayCommand]
    private async Task ShareVideoAsync()
    {
        if (string.IsNullOrEmpty(VideoPath) || !File.Exists(VideoPath))
        {
            await Shell.Current.DisplayAlert("Error", "Video file not found.", "OK");
            return;
        }

        await Share.Default.RequestAsync(new ShareFileRequest
        {
            Title = "Share Video",
            File = new ShareFile(VideoPath)
        });
    }

    [RelayCommand]
    private async Task SaveToGalleryAsync()
    {
        if (string.IsNullOrEmpty(VideoPath) || !File.Exists(VideoPath))
        {
            await Shell.Current.DisplayAlert("Error", "Video file not found.", "OK");
            return;
        }

        try
        {
            // On Android, copy to the public Movies directory
            var destDir = Android.OS.Environment.GetExternalStoragePublicDirectory(
                Android.OS.Environment.DirectoryMovies)?.AbsolutePath;

            if (destDir != null)
            {
                Directory.CreateDirectory(destDir);
                var destPath = Path.Combine(destDir, $"MobileI2V_{DateTime.Now:yyyyMMddHHmmss}.mp4");
                File.Copy(VideoPath, destPath, overwrite: true);

                // Notify media scanner
                var context = Android.App.Application.Context;
                Android.Media.MediaScannerConnection.ScanFile(
                    context, new[] { destPath }, null, null);

                await Shell.Current.DisplayAlert("Saved", 
                    "Video saved to Movies folder.", "OK");
            }
        }
        catch (Exception ex)
        {
            await Shell.Current.DisplayAlert("Error", 
                $"Failed to save: {ex.Message}", "OK");
        }
    }

    [RelayCommand]
    private async Task GenerateNewAsync()
    {
        await Shell.Current.GoToAsync("//home");
    }
}
