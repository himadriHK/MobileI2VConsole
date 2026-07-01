using MobileI2VConsole.Models;

namespace MobileI2VConsole.Tests.Models;

public class ModelRecordTests
{
    [Fact]
    public void GenerationRequest_HasDefaultValues()
    {
        var request = new GenerationRequest { ImagePath = "test.jpg" };
        Assert.Equal(1280, request.Width);
        Assert.Equal(720, request.Height);
        Assert.Equal(17, request.FrameCount);
        Assert.Equal(17, request.Fps);
        Assert.Equal(2, request.DiffusionSteps);
    }

    [Fact]
    public void GenerationResult_DefaultsToSuccess()
    {
        var result = new GenerationResult { VideoPath = "out.mp4" };
        Assert.True(result.Success);
        Assert.Null(result.ErrorMessage);
    }

    [Fact]
    public void GenerationProgress_TracksCorrectly()
    {
        var progress = new GenerationProgress
        {
            StepName = "Test",
            Progress = 0.5,
            OverallProgress = 0.25,
            CurrentFrame = 5,
            TotalFrames = 17
        };
        Assert.Equal(0.5, progress.Progress);
        Assert.Equal(5, progress.CurrentFrame);
        Assert.Equal(17, progress.TotalFrames);
    }

    [Fact]
    public void PromptTemplate_StoresValues()
    {
        var template = new PromptTemplate { Name = "Test", Prompt = "test prompt", Icon = "🔮" };
        Assert.Equal("Test", template.Name);
        Assert.Equal("test prompt", template.Prompt);
    }
}
