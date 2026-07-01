namespace MobileI2VConsole.Services;

/// <summary>
/// Wraps platform media picking functionality.
/// </summary>
public interface IMediaPickerService
{
    /// <summary>Opens the photo gallery for image selection.</summary>
    Task<FileResult?> PickImageAsync();

    /// <summary>Opens the camera to capture an image.</summary>
    Task<FileResult?> CaptureImageAsync();
}
