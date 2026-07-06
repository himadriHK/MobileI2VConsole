# Coding Conventions

Derived from actual source code in `src/MobileI2VConsole/` (C# .NET MAUI), `scripts/convert/` (Python), and `tests/` (C# xUnit).

---

## Language-Specific

### C# (.NET MAUI)

| Convention | Standard | Evidence |
|-----------|----------|----------|
| Language version | C# 12+ (implicit via `net10.0-android`) | `MobileI2VConsole.csproj` line 4 |
| Nullable reference types | **Enabled** (`<Nullable>enable</Nullable>`) | `MobileI2VConsole.csproj` line 10 |
| Implicit usings | **Enabled** (`<ImplicitUsings>enable</ImplicitUsings>`) | `MobileI2VConsole.csproj` line 9 |
| File-scoped namespaces | **Used consistently** — `namespace MobileI2VConsole.Services;` (no braces) | Every `.cs` file in `src/MobileI2VConsole/` |

**File-scoped namespace example** (`src/MobileI2VConsole/Services/FileService.cs` line 1):
```csharp
namespace MobileI2VConsole.Services;
```

**Nullable annotations** — `string?` for optional fields, `required` for mandatory record properties (`src/MobileI2VConsole/Models/GenerationRequest.cs` lines 9, 12):
```csharp
public required string ImagePath { get; init; }
public string? Prompt { get; init; }
```

### Python

| Convention | Standard | Evidence |
|-----------|----------|----------|
| Python version | 3.10+ (uses `from __future__ import annotations`) | `scripts/convert/common/model_utils.py` line 11 |
| Type hinting | **Used consistently** — `typing` module imports with `List`, `Dict`, `Optional`, `Sequence` | `scripts/convert/common/model_utils.py` line 17 |
| Docstring style | **Google-style** — triple-quoted docstrings with `Args:`, `Returns:` sections | `scripts/convert/common/model_utils.py` lines 66–82 |
| `__future__` annotations | **Used** — `from __future__ import annotations` at top of shared modules | `scripts/convert/common/model_utils.py` line 11, `scripts/convert/common/ort_config.py` line 10 |

**Type hinting example** (`scripts/convert/common/model_utils.py` line 66):
```python
def download_repo(
    repo_id: str,
    allow_patterns: Optional[List[str]] = None,
    local_dir: Optional[os.PathLike] = None,
) -> Path:
```

**Docstring example** (`scripts/convert/common/model_utils.py` lines 72–82):
```python
"""Download a HuggingFace repo snapshot (selective when ``allow_patterns`` is set).

Args:
    repo_id: HuggingFace repository ID (e.g. ``"hustvl/MobileI2V"``).
    allow_patterns: Glob patterns of files to include ...
    local_dir: Optional local directory to mirror the snapshot into.

Returns:
    Path to the local directory containing downloaded files.
"""
```

**Import organization** — stdlib → third-party → local, grouped with blank lines (`scripts/convert/convert_mobilei2v_unet.py` lines 13–38):
```python
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

import _pickle
import onnx
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from mobiledit import MobileditONNXWrapper, Attention, mobiledit_300m_P1_D16
from common.model_utils import (
    MODEL_CACHE_DIR,
    enable_cuda_optimizations,
    export_onnx,
    get_device,
    get_torch_dtype,
    verify_onnx,
)
```

---

## Naming Conventions

### C#

| Type | Convention | Example | Source File:Line |
|------|-----------|---------|-----------------|
| Classes | `PascalCase` | `HomeViewModel`, `FileService`, `InferenceOrchestrator` | `ViewModels/HomeViewModel.cs:10`, `Services/FileService.cs:6`, `Services/InferenceOrchestrator.cs:12` |
| Interfaces | `I` + `PascalCase` | `IFileService`, `IModelManager`, `IVideoEncoder` | `Services/IFileService.cs:6`, `Services/IModelManager.cs:8`, `Services/IVideoEncoder.cs:6` |
| Methods | `PascalCase` | `CreateMauiApp()`, `PickImageAsync()`, `GenerateVideoAsync()` | `MauiProgram.cs:12`, `ViewModels/HomeViewModel.cs:85`, `Services/InferenceOrchestrator.cs:43` |
| Private fields | `_camelCase` with underscore prefix | `_mediaPicker`, `_fileService`, `_modelManager`, `_cts` | `ViewModels/HomeViewModel.cs:12`, `Services/InferenceOrchestrator.cs:14` |
| Private constants | `PascalCase` | `VaeInputSize`, `OutputFrames`, `ProgressDiffusionStart` | `Services/InferenceOrchestrator.cs:18` |
| Local variables | `camelCase` | `result`, `templates`, `serialized` | `ViewModels/HomeViewModel.cs:48` |
| Public properties | `PascalCase` | `SelectedImagePath`, `PromptText`, `ImagePreview` | `ViewModels/HomeViewModel.cs:17` |
| Async methods | `Async` suffix | `PickImageAsync()`, `DownloadModelsAsync()`, `GenerateVideoAsync()` | `ViewModels/HomeViewModel.cs:85`, `Services/InferenceOrchestrator.cs:43` |
| Test classes | `{ClassName}Tests` | `HomeViewModelTests`, `FileServiceTests` | `tests/.../HomeViewModelTests.cs:8`, `tests/.../FileServiceTests.cs:5` |
| Test methods | `{Method}_{Scenario}` or `{Scenario}` | `Constructor_LoadsPromptTemplates`, `ApplyTemplate_SetsPromptText` | `tests/.../HomeViewModelTests.cs:22,29` |

**Private field pattern** (`src/MobileI2VConsole/ViewModels/HomeViewModel.cs` lines 12–14):
```csharp
private readonly IMediaPickerService _mediaPicker;
private readonly IFileService _fileService;
private readonly IModelManager _modelManager;
```

### Python

| Type | Convention | Example | Source File:Line |
|------|-----------|---------|-----------------|
| Modules/files | `snake_case.py` | `model_utils.py`, `ort_config.py`, `convert_mobilei2v_unet.py` | `scripts/convert/common/` |
| Functions | `snake_case` | `get_device()`, `export_onnx()`, `verify_onnx()` | `scripts/convert/common/model_utils.py:42,222,311` |
| Classes | `PascalCase` | `MobileditONNXWrapper`, `TurboVAEDDecoder3d` | `scripts/convert/models/mobiledit.py`, `scripts/convert/models/turbo_vaed_model.py` |
| Constants | `UPPER_SNAKE_CASE` | `MODEL_CACHE_DIR` | `scripts/convert/common/model_utils.py:39` |
| Private helpers | `_snake_case` with underscore prefix | `_patch_neg_transpose()`, `_torch_version_tuple()` | `scripts/convert/common/model_utils.py:193,156` |
| Test functions | `test_snake_case()` | `test_turbo_vaed_decoder_default_inject_noise_succeeds` | `scripts/convert/tests/test_turbo_vaed_bug.py:13` |

---

## Code Formatting

### C#

| Rule | Convention | Evidence |
|------|-----------|----------|
| Indentation | **Spaces**, 4-wide | All `.cs` files |
| Braces | **Allman style** (opening brace on new line) | Every class/method/control block |
| Line length | No hard limit observed (longest ~120 chars) | `InferenceOrchestrator.cs` line 94 |
| String quotes | **Double quotes** (`"`) for all strings | Throughout |
| `using` directives | **Inside namespace** (file-scoped), grouped: System → third-party → local | `ViewModels/HomeViewModel.cs` lines 1–6 |
| Blank lines | 1 blank line between `using` block and namespace, 1 between members | Throughout |
| `var` usage | **Used consistently** for local variables with obvious types | `var builder`, `var result`, `var templates` |

**Braces example** (`src/MobileI2VConsole/Services/InferenceOrchestrator.cs` lines 12–14):
```csharp
public class InferenceOrchestrator : IInferenceOrchestrator
{
```

### Python

| Rule | Convention | Evidence |
|------|-----------|----------|
| Indentation | **Spaces**, 4-wide | All `.py` files |
| Line length | ~100 chars (no hard limit observed) | `scripts/convert/common/model_utils.py` line 128 |
| String quotes | **Double quotes** (`"`) for docstrings and most strings; single quotes for short strings | Throughout |
| Blank lines | 2 blank lines between top-level functions, 1 between methods | `scripts/convert/common/model_utils.py` |
| Section separators | `# ──` comment blocks for visual section breaks | `scripts/convert/convert_mobilei2v_unet.py` lines 43, 60, 76, 146 |

**Section separator pattern** (`scripts/convert/convert_mobilei2v_unet.py` lines 43–45):
```python
# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------
```

---

## Module Patterns

### C# — MVVM Architecture

**ViewModels** inherit from `ObservableObject` (CommunityToolkit.Mvvm) and use source generators:

```csharp
// src/MobileI2VConsole/ViewModels/HomeViewModel.cs
public partial class HomeViewModel : ObservableObject
{
    [ObservableProperty]
    private string? selectedImagePath;   // generates public SelectedImagePath property

    [RelayCommand]
    private async Task PickImageAsync()  // generates PickImageCommand
    {
        ...
    }
}
```

**Services** follow interface/implementation separation with DI registration in `MauiProgram.cs`:

```csharp
// Interface
public interface IFileService { ... }

// Implementation
public class FileService : IFileService { ... }

// Registration (MauiProgram.cs)
builder.Services.AddSingleton<IFileService, FileService>();
```

**Models** are `record` types with `required` and `init` properties:

```csharp
// src/MobileI2VConsole/Models/GenerationRequest.cs
public record GenerationRequest
{
    public required string ImagePath { get; init; }
    public string? Prompt { get; init; }
    public int Width { get; init; } = 1280;
}
```

**Views** receive their ViewModel via constructor DI and set `BindingContext`:

```csharp
// src/MobileI2VConsole/Views/HomePage.xaml.cs
public partial class HomePage : ContentPage
{
    public HomePage(HomeViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }
}
```

**XAML** uses compiled bindings (`x:DataType`) and `clr-namespace` for ViewModel references:

```xml
<!-- src/MobileI2VConsole/Views/HomePage.xaml -->
<ContentPage xmlns:vm="clr-namespace:MobileI2VConsole.ViewModels"
             x:Class="MobileI2VConsole.Views.HomePage"
             x:DataType="vm:HomeViewModel">
```

**Shell navigation** — routes registered in `AppShell.xaml.cs`, navigation via `Shell.Current.GoToAsync()`:

```csharp
// AppShell.xaml.cs
Routing.RegisterRoute("generation", typeof(GenerationPage));

// HomeViewModel.cs
await Shell.Current.GoToAsync("generation", new Dictionary<string, object>
{
    ["Request"] = serialized
});
```

**Query properties** — ViewModels receive navigation parameters via `[QueryProperty]`:

```csharp
// GenerationViewModel.cs
[QueryProperty(nameof(Request), "Request")]
public partial class GenerationViewModel : ObservableObject
{
    partial void OnRequestChanged(string? value)
    {
        if (value != null) _ = StartGenerationAsync();
    }
}
```

### Python — Conversion Pipeline

**Package structure** — `__init__.py` files in each subdirectory with explicit exports:

```python
# scripts/convert/models/__init__.py
from .mobiledit import Mobiledit, mobiledit_300m_P1_D16
from .turbo_vaed_model import build_turbo_vaed_decoder, TurboVAEDDecoder3d
```

**Converter scripts** follow a consistent pattern:
1. `download_checkpoint()` — download model weights
2. `create_model()` — instantiate model architecture
3. `load_checkpoint()` — load weights into model
4. `export_to_onnx()` — wrap and export to ONNX
5. `main()` — CLI entry point with argparse
6. `convert_*()` — programmatic entry point for `convert_all.py`

**ONNX export** uses a shared `export_onnx()` utility from `common/model_utils.py` with `dynamo=False` for PyTorch 2.12+ compatibility.

---

## Error Handling

### C#

**Pattern: try/catch/finally with user-facing error messages** (`src/MobileI2VConsole/ViewModels/HomeViewModel.cs` lines 115–138):

```csharp
try
{
    var success = await _modelManager.DownloadModelsAsync(progress);
    if (success) { IsModelDownloaded = true; }
}
catch (Exception ex)
{
    DownloadStatusText = $"Download failed: {ex.Message}";
}
finally
{
    IsDownloading = false;
}
```

**Pattern: Guard clauses with user alerts** (`src/MobileI2VConsole/ViewModels/HomeViewModel.cs` lines 143–154):

```csharp
if (string.IsNullOrEmpty(SelectedImagePath))
{
    await Shell.Current.DisplayAlert("No Image", "Please select an image first.", "OK");
    return;
}
```

**Pattern: Specific exception types caught** (`src/MobileI2VConsole/Services/MediaPickerService.cs` lines 10–29):

```csharp
try { ... }
catch (FeatureNotSupportedException) { ... }
catch (PermissionException) { ... }
```

**Pattern: OperationCanceledException handled separately** (`src/MobileI2VConsole/ViewModels/GenerationViewModel.cs` lines 109–116):

```csharp
catch (OperationCanceledException)
{
    ErrorMessage = "Generation was cancelled";
}
catch (Exception ex)
{
    ErrorMessage = $"Generation failed: {ex.Message}";
}
```

**Pattern: Best-effort cleanup in finally** (`src/MobileI2VConsole/Services/InferenceOrchestrator.cs` lines 116–124):

```csharp
finally
{
    foreach (var model in new[] { "turbo_vaed", "mobilei2v_unet", "qwen2_encoder", "vae_encoder" })
    {
        try { await _modelManager.UnloadModelAsync(model); }
        catch { /* best-effort cleanup */ }
    }
}
```

**Pattern: Retry with backoff** (`src/MobileI2VConsole/Services/ModelManagerService.cs` lines 78–136):

```csharp
while (!success && attempt < maxRetries)
{
    attempt++;
    try { ... }
    catch
    {
        if (attempt >= maxRetries) throw;
        await Task.Delay(1000 * attempt, ct);
    }
}
```

### Python

**Pattern: try/except with logging** (`scripts/convert/common/model_utils.py` lines 137–148):

```python
try:
    hf_hub_download(...)
except Exception as exc:
    logger.warning("  Failed to download %s: %s", file_path, exc)
```

**Pattern: Graceful fallback** (`scripts/convert/convert_mobilei2v_unet.py` lines 98–106):

```python
try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
except _pickle.UnpicklingError:
    logger.warning("weights_only=True failed ...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

**Pattern: try/finally for state restoration** (`scripts/convert/convert_mobilei2v_unet.py` lines 220–232):

```python
Attention.forward = _memory_efficient_attn_forward
try:
    onnx_path = export_onnx(...)
finally:
    Attention.forward = _orig_attn_forward
```

---

## Async Patterns

### C#

| Pattern | Usage | Example |
|---------|-------|---------|
| `async Task` | All async methods return `Task` or `Task<T>` | `PickImageAsync()`, `GenerateVideoAsync()` |
| `await` | Used for all async calls | `await _mediaPicker.PickImageAsync()` |
| `Task.Run()` | CPU-bound work offloaded to thread pool | `InferenceOrchestrator.cs` lines 187, 217, 252, 300 |
| `CancellationToken` | Passed through pipeline for cancellation | `InferenceOrchestrator.cs` line 46 |
| `IProgress<T>` | Progress reporting pattern | `InferenceOrchestrator.cs` line 45 |
| `_ = StartGenerationAsync()` | Fire-and-forget from property change handler | `GenerationViewModel.cs` line 56 |

**Fire-and-forget pattern** (`src/MobileI2VConsole/ViewModels/GenerationViewModel.cs` lines 51–58):
```csharp
partial void OnRequestChanged(string? value)
{
    if (value != null)
    {
        _ = StartGenerationAsync();
    }
}
```

### Python

| Pattern | Usage | Example |
|---------|-------|---------|
| Synchronous | All Python code is synchronous | `scripts/convert/` |
| `torch.no_grad()` | Inference/export context manager | `scripts/convert/common/model_utils.py` line 271 |

---

## Testing Patterns

### C# (xUnit + Moq)

**Test structure** — Arrange/Act/Assert with `[Fact]` attributes (`tests/MobileI2VConsole.Tests/ViewModels/HomeViewModelTests.cs` lines 21–26):

```csharp
[Fact]
public void Constructor_LoadsPromptTemplates()
{
    var vm = new HomeViewModel(_mediaPickerMock.Object, _fileServiceMock.Object, _modelManagerMock.Object);
    Assert.NotEmpty(vm.PromptTemplates);
}
```

**Mock setup** — Moq with `Mock<T>` fields initialized in constructor (`tests/MobileI2VConsole.Tests/Services/ModelManagerServiceTests.cs` lines 9–15):

```csharp
private readonly Mock<IFileService> _fileServiceMock;

public ModelManagerServiceTests()
{
    _fileServiceMock = new Mock<IFileService>();
    _fileServiceMock.Setup(f => f.GetModelsDirectory()).Returns(Path.GetTempPath());
}
```

**Command execution** — ViewModel commands invoked via generated `ExecuteAsync` (`tests/MobileI2VConsole.Tests/ViewModels/HomeViewModelTests.cs` line 67):

```csharp
await vm.CheckModelStatusCommand.ExecuteAsync(null);
```

**Assertion style** — `Assert.Equal()`, `Assert.True()`, `Assert.False()`, `Assert.NotEmpty()`, `Assert.Contains()`, `Assert.NotNull()`, `Assert.EndsWith()`.

### Python (pytest)

**Test structure** — plain functions with `assert` statements (`scripts/convert/tests/test_turbo_vaed_bug.py` lines 13–25):

```python
def test_turbo_vaed_decoder_default_inject_noise_succeeds():
    """
    GIVEN the default inject_noise tuple has 5 elements (fixed)
    WHEN TurboVAEDDecoder3d is instantiated with all default arguments
    THEN no IndexError should be raised ...
    """
    decoder = TurboVAEDDecoder3d()
    assert isinstance(decoder, nn.Module)
    assert hasattr(decoder, "up_blocks")
    assert len(decoder.up_blocks) > 0
```

---

## Git Conventions

| Convention | Standard | Evidence |
|-----------|----------|----------|
| Commit messages | **Conventional commits** — `type: description` | `83cfc92 initial commit`, `57ba4cb new notebook` |
| Branch naming | No branches found (single `main` branch) | `git branch` shows only `main` |
| Commit frequency | 2 commits total (initial + notebook update) | `git log --oneline` |

---

## XAML Conventions

| Rule | Convention | Evidence |
|------|-----------|----------|
| Namespace declarations | `xmlns="http://schemas.microsoft.com/dotnet/2021/maui"` | All `.xaml` files |
| Compiled bindings | `x:DataType` on root element | `HomePage.xaml` line 6 |
| ViewModel namespace | `xmlns:vm="clr-namespace:MobileI2VConsole.ViewModels"` | `HomePage.xaml` line 4 |
| Color references | `{StaticResource ...}` for theme colors | `HomePage.xaml` line 8 |
| String resources | **Inline strings** (no `.resx` files found) | Throughout |
| Converter references | Registered in DI, used via `{Binding ..., Converter={x:Null}}` (placeholder) | `HomePage.xaml` lines 34, 40 |

**Note:** The XAML files use `{x:Null}` as converter references (e.g., `Converter={x:Null}` on lines 34, 40, 114, 120 of `HomePage.xaml`). This appears to be a placeholder — the actual converters (`InverseBoolConverter`, `NotNullToBoolConverter`) are registered in DI but not referenced by key in XAML. This is likely a work-in-progress state.

---

## Summary of Key Patterns

| Pattern | Where Used | Convention |
|---------|-----------|------------|
| MVVM with source generators | All ViewModels | `ObservableObject` + `[ObservableProperty]` + `[RelayCommand]` |
| Interface/Service DI | All services | `I{Name}` interface + `{Name}` implementation + DI registration |
| Record types for models | All Models | `record` with `required` + `init` properties |
| Constructor DI for pages | All Views | ViewModel injected, set as `BindingContext` |
| Shell navigation | AppShell | Routes registered, `GoToAsync()` with dictionary params |
| Progress reporting | Inference pipeline | `IProgress<T>` with `Progress<T>` callback |
| Cancellation support | Async operations | `CancellationToken` + `CancellationTokenSource` |
| Platform conditional compilation | Video encoder | `#if ANDROID` / `#else` in `MauiProgram.cs` |
| Python section separators | Converter scripts | `# ──` comment blocks |
| Python docstrings | Shared utilities | Google-style with `Args:` / `Returns:` |
| Python type hints | Shared utilities | `typing` module with `List`, `Dict`, `Optional` |
