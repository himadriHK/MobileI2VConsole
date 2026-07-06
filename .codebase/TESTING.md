# Testing Strategy

This project contains **two separate testing domains**: a .NET MAUI application (primary) and Python model conversion scripts (supporting). The testing infrastructure is minimal, with 17 tests total across both domains.

---

## .NET MAUI App (`MobileI2VConsole`)

### Test Framework

| Dependency | Version | Purpose |
|-----------|---------|---------|
| xUnit | 2.9.3 | Test framework |
| xunit.runner.visualstudio | 3.0.2 | Visual Studio test runner adapter |
| Moq | 4.20.72 | Mocking framework |
| coverlet.collector | 6.0.4 | Code coverage collector |

**Framework version evidence:** `tests/MobileI2VConsole.Tests/MobileI2VConsole.Tests.csproj`, lines 11-20.

### Test Location

- **Separate test project** at `tests/MobileI2VConsole.Tests/`
- Referenced via `ProjectReference` to `src/MobileI2VConsole/MobileI2VConsole.csproj` (line 24 of the .csproj)
- Project is registered in the solution file (`MobileI2VConsole.sln`, line 8)

### Test File Naming Convention

```
{ClassName}Tests.cs
```

Examples:
- `ModelRecordTests.cs`
- `FileServiceTests.cs`
- `ModelManagerServiceTests.cs`
- `HomeViewModelTests.cs`

### Test Organization

```
tests/MobileI2VConsole.Tests/
├── Models/
│   └── ModelRecordTests.cs          (4 tests)
├── Services/
│   ├── FileServiceTests.cs          (3 tests)
│   └── ModelManagerServiceTests.cs  (5 tests)
├── ViewModels/
│   └── HomeViewModelTests.cs        (4 tests)
├── MobileI2VConsole.Tests.csproj
├── bin/
└── obj/
```

The test directory structure mirrors the source structure under `src/MobileI2VConsole/`:
- `Models/` → tests for data models
- `Services/` → tests for service classes
- `ViewModels/` → tests for ViewModel classes

### Test Organization Patterns

**File-scoped namespaces** matching the source namespace pattern:
```
namespace MobileI2VConsole.Tests.Models;
namespace MobileI2VConsole.Tests.Services;
namespace MobileI2VConsole.Tests.ViewModels;
```

**No conftest files, no shared fixtures, no `[Collection]` attributes.** Test classes are entirely self-contained.

**No `xunit.runner.json` configuration file** found — tests run with xUnit defaults.

### Test Types

| Type | Count | Scope |
|------|-------|-------|
| Unit | 16 | Models (4), Services (8), ViewModels (4) |

There are **no integration tests, no E2E tests, no UI tests**. All tests are simple unit tests that validate individual class behavior.

### Test File Details

#### `Models/ModelRecordTests.cs` (4 tests)

Tests simple data model default values and property storage:
- `GenerationRequest_HasDefaultValues` — validates default dimensions, frame count, and steps
- `GenerationResult_DefaultsToSuccess` — validates default `Success = true` and null `ErrorMessage`
- `GenerationProgress_TracksCorrectly` — validates progress property assignment
- `PromptTemplate_StoresValues` — validates name/prompt storage

#### `Services/FileServiceTests.cs` (3 tests)

Tests `FileService` with real filesystem interaction (no mocking):
- `InitializeAsync_CreatesDirectories` — calls actual `InitializeAsync()` on Android-emulated context
- `GetFileSize_WhenFileDoesNotExist_ReturnsZero` — tests non-existent file path
- `CleanupTempAsync_DoesNotThrow` — smoke test that cleanup doesn't throw

**Note:** These tests interact with the real filesystem and use `Path.GetTempPath()`. `FileService` constructor takes no parameters (uses defaults for the platform), making it difficult to isolate.

#### `Services/ModelManagerServiceTests.cs` (5 tests)

Uses **Moq** to mock `IFileService` dependency:
- `Constructor_InitializesAllModelStatuses` — verifies 4 model statuses (`vae_encoder`, `qwen2_encoder`, `mobilei2v_unet`, `turbo_vaed`)
- `IsModelDownloaded_ReturnsFalseInitially` — verify initial state
- `IsModelLoaded_ReturnsFalseInitially` — verify initial state
- `GetModelPath_ReturnsExpectedPath` — verify path suffix `.onnx`
- `LoadModelAsync_WhenFileNotFound_ReturnsFalse` — test failure path with nonexistent directory

#### `ViewModels/HomeViewModelTests.cs` (4 tests)

Uses **Moq** to mock `IMediaPickerService`, `IFileService`, and `IModelManager`:
- `Constructor_LoadsPromptTemplates` — validates templates are loaded
- `ApplyTemplate_SetsPromptText` — validates command sets prompt text
- `SelectedImageChanged_UpdatesImagePreview` — validates image preview is set (writes to real temp dir)
- `CheckModelStatus_UpdatesFlag` — validates `IsModelDownloaded` flag with mock statuses

### Coverage

**No coverage data available.** While `coverlet.collector` 6.0.4 is referenced in the test project (line 17 of `MobileI2VConsole.Tests.csproj`), there is no evidence that coverage has ever been run or that a coverage threshold is enforced. No coverage reports exist in the repository.

### Running Tests

```bash
# Via dotnet CLI (requires .NET 10 SDK with Android workload)
dotnet test tests/MobileI2VConsole.Tests/MobileI2VConsole.Tests.csproj

# With coverage
dotnet test tests/MobileI2VConsole.Tests/MobileI2VConsole.Tests.csproj --collect:"XPlat Code Coverage"

# Via Visual Studio Test Explorer
# Open MobileI2VConsole.sln → Test → Test Explorer
```

**Note:** The test project targets `net10.0-android36.0`, which requires the .NET 10 SDK and the Android workload. Running tests outside of that environment may fail.

---

## Python Model Conversion Scripts

### Test Framework

**pytest** is used (inferred from `__pycache__` containing `pytest-9.1.1` at `scripts/convert/tests/__pycache__/test_turbo_vaed_bug.cpython-312-pytest-9.1.1.pyc`).

No explicit pytest configuration files exist (no `pytest.ini`, `setup.cfg`, or `pyproject.toml` with pytest config). No test dependencies are declared in `scripts/convert/requirements.txt`.

### Test Location

- **Separate test directory** at `scripts/convert/tests/`
- Contains `__init__.py` making it a package
- Tests import from `models.turbo_vaed_model` (located at `scripts/convert/models/turbo_vaed_model.py`)

### Test File Naming Convention

```
test_{feature}.py
```

Example: `test_turbo_vaed_bug.py`

### Test Files

#### `tests/test_turbo_vaed_bug.py` (1 test)

A regression test for the `TurboVAEDDecoder3d` class:

```python
def test_turbo_vaed_decoder_default_inject_noise_succeeds():
    """
    GIVEN the default inject_noise tuple has 5 elements (fixed)
    WHEN TurboVAEDDecoder3d is instantiated with all default arguments
    THEN no IndexError should be raised and the decoder should have valid structure.
    """
    decoder = TurboVAEDDecoder3d()
    assert isinstance(decoder, nn.Module)
    assert hasattr(decoder, "up_blocks")
    assert len(decoder.up_blocks) > 0
```

Key patterns:
- Module-level docstring explaining the regression context (lines 1-7)
- GIVEN/WHEN/THEN docstring in the test function (lines 14-18)
- Plain `assert` statements (no pytest-specific `assert_*` helpers)
- Depends on `torch.nn` and the local model module

### Running Tests

```bash
# From the scripts/convert/ directory
python -m pytest tests/

# With verbose output
python -m pytest tests/ -v
```

Requires PyTorch (`torch>=2.1.0`) and the `diffusers` library (see `scripts/convert/requirements.txt`).

---

## CI Integration

**No CI/CD configuration exists.** There are no GitHub Actions workflows, Azure Pipelines, or any other CI configuration files in the repository. Tests must be run manually.

---

## Summary

### Test Inventory

| Domain | Framework | Test Files | Test Methods | Coverage |
|--------|-----------|------------|-------------|----------|
| .NET MAUI App | xUnit 2.9.3 + Moq | 4 | 16 | Not measured |
| Python Scripts | pytest | 1 | 1 | Not measured |
| **Total** | — | **5** | **17** | **No coverage data** |

### Strengths

1. Test project structure mirrors source structure (Models/Services/ViewModels)
2. Dependency injection is used in the main app (CommunityToolkit.Mvvm), making services mockable
3. Moq is used appropriately in `ModelManagerServiceTests` and `HomeViewModelTests`
4. The Python test follows a clear GIVEN/WHEN/THEN pattern with regression context documented

### Gaps

1. **No integration or E2E tests** — no tests exercise the full app flow or ONNX model inference path
2. **No parametrized tests** — `[Theory]` (xUnit) or `@pytest.mark.parametrize` are not used anywhere
3. **No `xunit.runner.json`** — test execution is entirely on xUnit defaults
4. **No CI/CD pipeline** — no automated test execution
5. **No coverage threshold** — `coverlet.collector` is referenced but not enforced
6. **FileServiceTests interact with real filesystem** — no filesystem abstraction or temp directory isolation
7. **Python tests are minimal** — only 1 regression test covering a single class
8. **No async test patterns despite async code** — the app uses async methods but tests use `.Result` for blocking (e.g., `HomeViewModelTests` line 67 uses `ExecuteAsync`)
