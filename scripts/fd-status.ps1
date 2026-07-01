#!/usr/bin/env pwsh
# fd-status.ps1 — FlowDeck Planning Status
# Reads from .planning/ directory to show project status, roadmap, workspace overview, or phase details.

[CmdletBinding()]
param(
    [switch]$Roadmap,
    [switch]$Workspace,
    [string]$Phase
)

# ── Support --flag=value format in addition to PowerShell named params ──
foreach ($arg in $args) {
    if ($arg -match '^--phase[=:](\d+)$') {
        $Phase = $matches[1]
    } elseif ($arg -match '^--roadmap$') {
        $Roadmap = $true
    } elseif ($arg -match '^--workspace$') {
        $Workspace = $true
    }
}

# ── Ensure UTF-8 output for box-drawing characters ──────────────────────
$prevEncoding = [Console]::OutputEncoding
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}

# ── Determine project root ──────────────────────────────────────────────
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path 2>$null
    if (-not $scriptDir) { $scriptDir = Get-Location }
}
$projectRoot = Resolve-Path "$scriptDir/.."
$planningDir = Join-Path $projectRoot ".planning"
$stateFile    = Join-Path $planningDir "STATE.md"

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

function Parse-YamlFrontmatter {
    <#
    .SYNOPSIS
        Parse YAML frontmatter (between --- markers) from a Markdown file.
    #>
    param([string]$FilePath)
    if (-not (Test-Path -LiteralPath $FilePath)) { return $null }

    $text = Get-Content -Path $FilePath -Raw
    if (-not $text) { return $null }

    # Match opening ---, content, closing ---
    if ($text -notmatch '(?ms)^---\s*\n(.*?)\n---\s*\n') {
        return $null
    }

    $yamlBlock = $matches[1]
    $result = @{}

    foreach ($line in $yamlBlock -split "`n") {
        $line = $line.Trim()
        if (-not $line -or $line -match '^#') { continue }
        if ($line -match '^(\w[\w_-]*)\s*:\s*(.*)') {
            $key   = $matches[1]
            $value = $matches[2].Trim()
            # Strip surrounding quotes
            if ($value.Length -ge 2) {
                if (($value[0] -eq '"' -and $value[-1] -eq '"') -or
                    ($value[0] -eq "'" -and $value[-1] -eq "'")) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            $result[$key] = $value
        }
    }

    return $result
}

function Parse-ArrayField {
    <#
    .SYNOPSIS
        Parse a YAML array string like "[1, 2, 3]" or "[]" into a list of values.
    #>
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value) -or $Value.Trim() -eq '[]') {
        return @()
    }
    $cleaned = $Value.Trim() -replace '^\[|\]$', '' -replace '\s', ''
    if ([string]::IsNullOrEmpty($cleaned)) { return @() }
    return ,($cleaned -split ',')
}

function Get-StateData {
    <#
    .SYNOPSIS
        Read and parse STATE.md frontmatter into a hashtable.
    #>
    $fm = Parse-YamlFrontmatter -FilePath $stateFile
    if (-not $fm) { return $null }

    return @{
        Phase         = $fm['phase']
        Status        = $fm['status']
        PlanConfirmed = $fm['plan_confirmed']
        StepsComplete = Parse-ArrayField $fm['steps_complete']
        StepsPending  = Parse-ArrayField $fm['steps_pending']
        LastUpdated   = $fm['lastUpdatedAt']
        LastAction    = $fm['last_action']
        NextAction    = $fm['next_action']
    }
}

function Format-Timestamp {
    param([string]$Raw)
    if ([string]::IsNullOrEmpty($Raw)) { return '-' }
    try {
        $dt = [DateTime]::Parse($Raw)
        return $dt.ToString('yyyy-MM-dd HH:mm:ss')
    } catch {
        return $Raw
    }
}

function Test-Yes {
    param([string]$Value)
    return ($Value -eq 'true' -or $Value -eq 'yes' -or $Value -eq $true)
}

function Get-StepNamesFromPlan {
    <#
    .SYNOPSIS
        Extract step names from PLAN.md headings.
        Returns a hashtable: { 1 = "Name", 2 = "Name", ... }
    #>
    param([string]$PlanFilePath)
    if (-not (Test-Path -LiteralPath $PlanFilePath)) { return @{} }

    $content = Get-Content -Path $PlanFilePath -Raw
    if (-not $content) { return @{} }

    $stepNames = @{}

    # 1) Match headings: ## or ### optional (Step|Wave|Phase) N: Name
    $headingRegex = [regex]::new(
        '^#{2,4}\s+(?:(?:Step|Wave|Phase)\s+)?(\d+)\s*[:\-–—]\s*(.+?)$',
        [System.Text.RegularExpressions.RegexOptions]::Multiline
    )
    $matches = $headingRegex.Matches($content)
    foreach ($m in $matches) {
        $num  = [int]$m.Groups[1].Value
        $name = $m.Groups[2].Value.Trim()
        if (-not $stepNames.ContainsKey($num)) {
            $stepNames[$num] = $name
        }
    }

    # 2) Fallback: numbered list items (1.  xxx) if not already found
    if ($stepNames.Count -eq 0) {
        $itemRegex = [regex]::new(
            '^\d+\.\s+(.+?)$',
            [System.Text.RegularExpressions.RegexOptions]::Multiline
        )
        $itemMatches = $itemRegex.Matches($content)
        $idx = 1
        foreach ($m in $itemMatches) {
            $stepNames[$idx] = $m.Groups[1].Value.Trim()
            $idx++
        }
    }

    return $stepNames
}

# ═══════════════════════════════════════════════════════════════════════
# MODE 1: DEFAULT (no flags)
# ═══════════════════════════════════════════════════════════════════════

function Show-DefaultStatus {
    if (-not (Test-Path -LiteralPath $stateFile)) {
        Write-Host 'No active workspace. Run "/fd-map-codebase" to initialize, then "/fd-new-feature" to start a feature.'
        return
    }

    $state = Get-StateData
    if (-not $state) {
        Write-Host 'Could not parse STATE.md.'
        return
    }

    $totalSteps  = $state.StepsComplete.Count + $state.StepsPending.Count
    $completed   = $state.StepsComplete.Count
    $confirmed   = if (Test-Yes $state.PlanConfirmed) { 'yes' } else { 'no' }

    Write-Host ("═" * 56)
    Write-Host ('Phase: {0}  |  Status: {1}  |  Updated: {2}' -f $state.Phase, $state.Status, (Format-Timestamp $state.LastUpdated))
    Write-Host ("─" * 56)
    Write-Host ('Plan: {0} steps ({1} complete)' -f $totalSteps, $completed)
    Write-Host ('Plan confirmed: {0}' -f $confirmed)
    Write-Host ("═" * 56)
}

# ═══════════════════════════════════════════════════════════════════════
# MODE 2: --roadmap
# ═══════════════════════════════════════════════════════════════════════

function Show-Roadmap {
    $roadmapFile = Join-Path $planningDir 'ROADMAP.md'
    $phasesDir   = Join-Path $planningDir 'phases'

    $state       = Get-StateData
    $currentPhase = if ($state -and $state.Phase) { [int]$state.Phase } else { $null }

    # ── Try reading ROADMAP.md ──
    if (Test-Path -LiteralPath $roadmapFile) {
        $content = Get-Content -Path $roadmapFile -Raw
        $phases  = @()

        $phaseRegex = [regex]::new(
            '^##\s+Phase\s+(\d+)\s*[:\-–—]\s*(.+?)$',
            [System.Text.RegularExpressions.RegexOptions]::Multiline
        )
        $matches = $phaseRegex.Matches($content)
        foreach ($m in $matches) {
            $phases += @{
                Number = [int]$m.Groups[1].Value
                Name   = $m.Groups[2].Value.Trim()
            }
        }

        if ($phases.Count -eq 0) {
            Write-Host 'No roadmap data found.'
            return
        }

        Write-Host ("═" * 39)
        Write-Host 'PROJECT ROADMAP'
        Write-Host ("═" * 39)

        $sorted = $phases | Sort-Object Number
        foreach ($p in $sorted) {
            if ($currentPhase -and $p.Number -lt $currentPhase) {
                Write-Host ('  ✅ Phase {0}: {1} — completed' -f $p.Number, $p.Name)
            } elseif ($currentPhase -and $p.Number -eq $currentPhase) {
                $phaseStatus = if ($state) { $state.Status.ToLower() } else { '' }
                if ($phaseStatus -in 'complete', 'completed', 'done') {
                    Write-Host ('  ✅ Phase {0}: {1} — completed' -f $p.Number, $p.Name)
                } else {
                    Write-Host ('  🔄 Phase {0}: {1} — in progress  ← current' -f $p.Number, $p.Name)
                }
            } else {
                Write-Host ('  ⏳ Phase {0}: {1} — planned' -f $p.Number, $p.Name)
            }
        }
        Write-Host ("═" * 39)
        return
    }

    # ── ROADMAP.md doesn't exist — infer from phases/ directories ──
    if (Test-Path -LiteralPath $phasesDir) {
        $phaseDirs = Get-ChildItem -Path $phasesDir -Directory | Sort-Object Name
    } else {
        $phaseDirs = @()
    }

    if ($phaseDirs.Count -eq 0) {
        Write-Host 'No roadmap data found.'
        return
    }

    Write-Host ("═" * 39)
    Write-Host 'PROJECT ROADMAP'
    Write-Host ("═" * 39)

    foreach ($dir in $phaseDirs) {
        if ($dir.Name -match 'phase-(\d+)') {
            $num = [int]$matches[1]
            $name = $dir.Name
            # Try to get a friendly name from the phase's PLAN.md
            $planFile = Join-Path $dir.FullName 'PLAN.md'
            if (Test-Path -LiteralPath $planFile) {
                $firstLine = Get-Content -Path $planFile -TotalCount 1
                if ($firstLine -match '^#\s+Phase\s+\d+\s*[:\-–—]\s*(.+)') {
                    $name = $matches[1].Trim()
                }
            }

            if ($currentPhase -and $num -lt $currentPhase) {
                Write-Host ('  ✅ Phase {0}: {1} — completed' -f $num, $name)
            } elseif ($currentPhase -and $num -eq $currentPhase) {
                $phaseStatus = if ($state) { $state.Status.ToLower() } else { '' }
                if ($phaseStatus -in 'complete', 'completed', 'done') {
                    Write-Host ('  ✅ Phase {0}: {1} — completed' -f $num, $name)
                } else {
                    Write-Host ('  🔄 Phase {0}: {1} — in progress  ← current' -f $num, $name)
                }
            } else {
                Write-Host ('  ⏳ Phase {0}: {1} — planned' -f $num, $name)
            }
        }
    }
    Write-Host ("═" * 39)
}

# ═══════════════════════════════════════════════════════════════════════
# MODE 3: --workspace
# ═══════════════════════════════════════════════════════════════════════

function Show-Workspace {
    $configFile = Join-Path $planningDir 'config.json'

    Write-Host ("═" * 56)
    Write-Host 'WORKSPACE OVERVIEW'
    Write-Host ("═" * 56)

    # ── Try reading config.json ──
    if (Test-Path -LiteralPath $configFile) {
        $config = Get-Content -Path $configFile -Raw | ConvertFrom-Json
        if ($config.repos -and @($config.repos).Count -gt 0) {
            $totalRepos    = @($config.repos).Count
            $inProgress    = 0
            $completed     = 0
            $planned       = 0

            foreach ($repo in $config.repos) {
                $repoName = $repo.name
                $repoPath = $repo.path

                # Resolve relative to project root
                if (-not [System.IO.Path]::IsPathRooted($repoPath)) {
                    $resolved = Join-Path $projectRoot $repoPath
                } else {
                    $resolved = $repoPath
                }

                $repoStateFile = Join-Path $resolved '.planning' 'STATE.md'
                $phaseDisplay   = '-'
                $statusDisplay  = '-'
                $planDisplay    = '❌'
                $updatedDisplay = '-'

                if (Test-Path -LiteralPath $repoStateFile) {
                    $rs = Parse-YamlFrontmatter -FilePath $repoStateFile
                    if ($rs) {
                        $phaseDisplay   = $rs['phase']
                        $statusDisplay  = $rs['status']
                        $planDisplay    = if (Test-Yes $rs['plan_confirmed']) { '✅' } else { '❌' }
                        $updatedDisplay = Format-Timestamp $rs['lastUpdatedAt']

                        $s = if ($rs['status']) { $rs['status'].ToLower() } else { '' }
                        if ($s -eq 'in_progress' -or $s -eq 'active') {
                            $inProgress++
                        } elseif ($s -eq 'complete' -or $s -eq 'done' -or $s -eq 'completed') {
                            $completed++
                        } else {
                            $planned++
                        }
                    } else {
                        $planned++
                    }
                } else {
                    $planned++
                }

                Write-Host ('  {0,-20} — Phase {1,-3} | {2,-12} | Plan: {3} | Updated: {4}' -f
                    $repoName, $phaseDisplay, $statusDisplay, $planDisplay, $updatedDisplay)
            }

            Write-Host ("─" * 56)
            Write-Host ('Total: {0} repos | {1} in progress | {2} completed | {3} planned' -f $totalRepos, $inProgress, $completed, $planned)
            Write-Host ("═" * 56)
            return
        }
    }

    # ── Fallback: no repos configured — show current project only ──
    Write-Host '  No workspace repos are configured. Showing current project:'
    Write-Host ''

    $state = Get-StateData
    if ($state) {
        $planDisplay    = if (Test-Yes $state.PlanConfirmed) { '✅' } else { '❌' }
        $updatedDisplay = Format-Timestamp $state.LastUpdated
        $projName       = (Get-Item $projectRoot).Name
        Write-Host ('  {0} — Phase {1} | {2} | Plan: {3} | Updated: {4}' -f
            $projName, $state.Phase, $state.Status, $planDisplay, $updatedDisplay)
    } else {
        Write-Host '  (No STATE.md found for current project)'
    }
    Write-Host ("═" * 56)
}

# ═══════════════════════════════════════════════════════════════════════
# MODE 4: --phase=N
# ═══════════════════════════════════════════════════════════════════════

function Show-PhaseDetail {
    param([int]$PhaseNum)

    if (-not (Test-Path -LiteralPath $stateFile)) {
        Write-Host 'No active workspace. Run "/fd-map-codebase" to initialize, then "/fd-new-feature" to start a feature.'
        return
    }

    $phaseDir = Join-Path $planningDir 'phases' "phase-$PhaseNum"
    if (-not (Test-Path -LiteralPath $phaseDir)) {
        Write-Host "Phase $PhaseNum not found."
        return
    }

    $planFile = Join-Path $phaseDir 'PLAN.md'
    if (-not (Test-Path -LiteralPath $planFile)) {
        Write-Host "Phase $PhaseNum PLAN.md not found."
        return
    }

    $state = Get-StateData

    Write-Host ("═" * 56)
    Write-Host "PHASE $PhaseNum DETAIL"
    Write-Host ("═" * 56)

    if ($state) {
        $confirmed = if (Test-Yes $state.PlanConfirmed) { 'yes' } else { 'no' }
        Write-Host ('Status: {0}' -f $state.Status)
        Write-Host ('Plan file: {0}' -f $planFile)
        Write-Host ('Plan confirmed: {0}' -f $confirmed)
        Write-Host ''
        Write-Host 'Steps:'

        # ── Build step-name mapping from PLAN.md ──
        $stepNames = Get-StepNamesFromPlan -PlanFilePath $planFile

        # Build lookup sets
        $completeSet = @{}
        foreach ($s in $state.StepsComplete) { $completeSet[[int]$s] = $true }

        $pendingSet = @{}
        foreach ($s in $state.StepsPending) { $pendingSet[[int]$s] = $true }

        # Determine the maximum step number to display
        $maxStep = 0
        foreach ($s in $state.StepsComplete) { if ([int]$s -gt $maxStep) { $maxStep = [int]$s } }
        foreach ($s in $state.StepsPending)  { if ([int]$s -gt $maxStep) { $maxStep = [int]$s } }
        foreach ($k in $stepNames.Keys)      { if ($k -gt $maxStep) { $maxStep = $k } }

        if ($maxStep -eq 0) {
            Write-Host '  (No step information available)'
        } else {
            for ($i = 1; $i -le $maxStep; $i++) {
                $name = if ($stepNames.ContainsKey($i)) { $stepNames[$i] } else { '' }
                if ($completeSet.ContainsKey($i)) {
                    if ($name) { Write-Host ('  ✅ Step {0}: {1} — completed' -f $i, $name) }
                    else       { Write-Host ('  ✅ Step {0} — completed' -f $i) }
                } elseif ($pendingSet.ContainsKey($i)) {
                    if ($name) { Write-Host ('  ⬜ Step {0}: {1} — pending' -f $i, $name) }
                    else       { Write-Host ('  ⬜ Step {0} — pending' -f $i) }
                } else {
                    # Step exists in PLAN.md names but not in STATE arrays
                    if ($name) { Write-Host ('  ⬜ Step {0}: {1} — pending' -f $i, $name) }
                }
            }
        }
    } else {
        Write-Host 'Status: unknown'
        Write-Host ('Plan file: {0}' -f $planFile)
        Write-Host ''
        Write-Host 'Steps:'
        Write-Host '  (State data could not be parsed)'
    }

    Write-Host ("═" * 56)
}

# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ═══════════════════════════════════════════════════════════════════════

if (-not (Test-Path -LiteralPath $planningDir)) {
    Write-Host 'No active workspace. Run "/fd-map-codebase" to initialize, then "/fd-new-feature" to start a feature.'
    exit 1
}

if ($Roadmap) {
    Show-Roadmap
} elseif ($Workspace) {
    Show-Workspace
} elseif ($PSBoundParameters.ContainsKey('Phase') -or (-not [string]::IsNullOrEmpty($Phase))) {
    $phaseNum = 0
    if (-not [int]::TryParse($Phase, [ref]$phaseNum)) {
        Write-Host "Invalid phase number: $Phase"
        exit 1
    }
    Show-PhaseDetail -PhaseNum $phaseNum
} else {
    Show-DefaultStatus
}
