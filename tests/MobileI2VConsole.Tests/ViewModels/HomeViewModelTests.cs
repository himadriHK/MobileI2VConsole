using MobileI2VConsole.Models;
using MobileI2VConsole.Services;
using MobileI2VConsole.ViewModels;
using Moq;

namespace MobileI2VConsole.Tests.ViewModels;

public class HomeViewModelTests
{
    private readonly Mock<IMediaPickerService> _mediaPickerMock;
    private readonly Mock<IFileService> _fileServiceMock;
    private readonly Mock<IModelManager> _modelManagerMock;

    public HomeViewModelTests()
    {
        _mediaPickerMock = new Mock<IMediaPickerService>();
        _fileServiceMock = new Mock<IFileService>();
        _modelManagerMock = new Mock<IModelManager>();
    }

    [Fact]
    public void Constructor_LoadsPromptTemplates()
    {
        var vm = new HomeViewModel(_mediaPickerMock.Object, _fileServiceMock.Object, _modelManagerMock.Object);
        Assert.NotEmpty(vm.PromptTemplates);
    }

    [Fact]
    public void ApplyTemplate_SetsPromptText()
    {
        var vm = new HomeViewModel(_mediaPickerMock.Object, _fileServiceMock.Object, _modelManagerMock.Object);
        var template = new PromptTemplate { Name = "Test", Prompt = "test prompt", Icon = "🔮" };
        vm.ApplyTemplateCommand.Execute(template);
        Assert.Equal("test prompt", vm.PromptText);
    }

    [Fact]
    public void SelectedImageChanged_UpdatesImagePreview()
    {
        var vm = new HomeViewModel(_mediaPickerMock.Object, _fileServiceMock.Object, _modelManagerMock.Object);
        var testPath = Path.GetTempFileName();
        try
        {
            File.WriteAllText(testPath, "fake image");
            vm.SelectedImagePath = testPath;
            Assert.NotNull(vm.ImagePreview);
        }
        finally
        {
            File.Delete(testPath);
        }
    }

    [Fact]
    public async Task CheckModelStatus_UpdatesFlag()
    {
        var statuses = new List<ModelStatus>
        {
            new ModelStatus { ModelName = "vae_encoder", IsDownloaded = true, DownloadProgress = 1.0 },
            new ModelStatus { ModelName = "qwen2_encoder", IsDownloaded = true, DownloadProgress = 1.0 },
            new ModelStatus { ModelName = "mobilei2v_unet", IsDownloaded = true, DownloadProgress = 1.0 },
            new ModelStatus { ModelName = "turbo_vaed", IsDownloaded = true, DownloadProgress = 1.0 }
        };
        _modelManagerMock.Setup(m => m.GetAllStatus()).Returns(statuses);

        var vm = new HomeViewModel(_mediaPickerMock.Object, _fileServiceMock.Object, _modelManagerMock.Object);
        await vm.CheckModelStatusCommand.ExecuteAsync(null);
        Assert.True(vm.IsModelDownloaded);
    }
}
