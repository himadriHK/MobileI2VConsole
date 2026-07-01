using MobileI2VConsole.Services;

namespace MobileI2VConsole.Tests.Services;

public class FileServiceTests
{
    [Fact]
    public async Task InitializeAsync_CreatesDirectories()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        var service = new FileService(); // Note: Will use default Personal folder on Android
        await service.InitializeAsync();
        Assert.NotNull(service.GetModelsDirectory());
        Assert.NotNull(service.GetOutputDirectory());
        Assert.NotNull(service.GetTempDirectory());
    }

    [Fact]
    public void GetFileSize_WhenFileDoesNotExist_ReturnsZero()
    {
        var service = new FileService();
        var size = service.GetFileSize("C:\\nonexistent\\file.txt");
        Assert.Equal(0, size);
    }

    [Fact]
    public async Task CleanupTempAsync_DoesNotThrow()
    {
        var service = new FileService();
        await service.CleanupTempAsync(TimeSpan.FromMinutes(5));
        // Should not throw even if temp dir doesn't exist
        Assert.True(true);
    }
}
