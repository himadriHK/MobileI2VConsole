## 2026-07-01 — FlowDeck orchestrator guard on Windows
**Severity:** medium
**Mistake:** Used Test-Path and Get-Item which were blocked by the orchestrator guard even though they are read-only operations on non-sensitive paths
**Lesson:** On Windows, the orchestrator guard restricts PowerShell cmdlets like Test-Path and Get-Item even for benign file checks. Use `read` (directory listing) or `bash ls` for file existence checks instead. fdx-ls is preferred but may not be in PATH on Windows.
