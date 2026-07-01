namespace MobileI2VConsole.Services;

/// <summary>
/// Wraps MAUI's built-in <see cref="MediaPicker"/> for image selection and capture.
/// </summary>
public class MediaPickerService : IMediaPickerService
{
    public async Task<FileResult?> PickImageAsync()
    {
        try
        {
            var result = await MediaPicker.Default.PickPhotoAsync(new MediaPickerOptions
            {
                Title = "Select an image to animate"
            });
            return result;
        }
        catch (FeatureNotSupportedException)
        {
            await Shell.Current.DisplayAlert("Not Supported",
                "Photo picking is not supported on this device.", "OK");
            return null;
        }
        catch (PermissionException)
        {
            await Shell.Current.DisplayAlert("Permission Denied",
                "Gallery permission is required to select images.", "OK");
            return null;
        }
    }

    public async Task<FileResult?> CaptureImageAsync()
    {
        try
        {
            if (!MediaPicker.Default.IsCaptureSupported)
            {
                await Shell.Current.DisplayAlert("Not Available",
                    "Camera capture is not available on this device.", "OK");
                return null;
            }

            var result = await MediaPicker.Default.CapturePhotoAsync(new MediaPickerOptions
            {
                Title = "Take a photo to animate"
            });
            return result;
        }
        catch (FeatureNotSupportedException)
        {
            await Shell.Current.DisplayAlert("Not Supported",
                "Camera is not supported on this device.", "OK");
            return null;
        }
        catch (PermissionException)
        {
            await Shell.Current.DisplayAlert("Permission Denied",
                "Camera permission is required to capture photos.", "OK");
            return null;
        }
    }
}
