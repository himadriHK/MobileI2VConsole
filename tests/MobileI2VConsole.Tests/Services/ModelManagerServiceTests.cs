using MobileI2VConsole.Models;
using MobileI2VConsole.Services;
using Moq;

namespace MobileI2VConsole.Tests.Services;

public class ModelManagerServiceTests
{
    private readonly Mock<IFileService> _fileServiceMock;

    public ModelManagerServiceTests()
    {
        _fileServiceMock = new Mock<IFileService>();
        _fileServiceMock.Setup(f => f.GetModelsDirectory()).Returns(Path.GetTempPath());
    }

    [Fact]
    public void Constructor_InitializesAllModelStatuses()
    {
        var manager = new ModelManagerService(_fileServiceMock.Object);
        var statuses = manager.GetAllStatus();
        Assert.Equal(4, statuses.Count);
        Assert.Contains(statuses, s => s.ModelName == "vae_encoder");
        Assert.Contains(statuses, s => s.ModelName == "qwen2_encoder");
        Assert.Contains(statuses, s => s.ModelName == "mobilei2v_unet");
        Assert.Contains(statuses, s => s.ModelName == "turbo_vaed");
    }

    [Fact]
    public void IsModelDownloaded_ReturnsFalseInitially()
    {
        var manager = new ModelManagerService(_fileServiceMock.Object);
        Assert.False(manager.IsModelDownloaded("vae_encoder"));
    }

    [Fact]
    public void IsModelLoaded_ReturnsFalseInitially()
    {
        var manager = new ModelManagerService(_fileServiceMock.Object);
        Assert.False(manager.IsModelLoaded("vae_encoder"));
    }

    [Fact]
    public void GetModelPath_ReturnsExpectedPath()
    {
        var manager = new ModelManagerService(_fileServiceMock.Object);
        var path = manager.GetModelPath("vae_encoder");
        Assert.EndsWith("vae_encoder.onnx", path);
    }

    [Fact]
    public void LoadModelAsync_WhenFileNotFound_ReturnsFalse()
    {
        _fileServiceMock.Setup(f => f.GetModelsDirectory()).Returns("C:\\nonexistent\\path");
        var manager = new ModelManagerService(_fileServiceMock.Object);
        var result = manager.LoadModelAsync("vae_encoder", CancellationToken.None).Result;
        Assert.False(result);
    }
}
