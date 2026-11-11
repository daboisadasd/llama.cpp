# Native Windows Phase 1 launcher: RTX 5080 target + RTX 3070 Ti eGPU native MTP.
# Use Developer PowerShell for Visual Studio 2022. No WSL or Bash.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Check','Build','Download','Test','Smoke','Full','Profile','Update')]
    [string]$Action,
    [string]$ModelDir = (Join-Path (Join-Path $env:USERPROFILE 'models') 'Qwen3.8-27B'),
    [string]$PrimaryDevice = 'CUDA0',
    [string]$DraftDevice = 'CUDA1',
    [ValidateRange(0,256)] [int]$TargetNgl = 35,
    [string]$ExpectPrimary = '5080',
    [string]$ExpectDraft = '3070 Ti',
    [string]$BuildDir = '',
    [string]$NvidiaSmi = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$toolDir = Split-Path -Parent $PSCommandPath
$repoRoot = (Resolve-Path (Join-Path $toolDir '..\..')).Path
if ([string]::IsNullOrWhiteSpace($BuildDir)) { $BuildDir = Join-Path $repoRoot 'build-win' }
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' not found. See tools/phase1_mtp/WINDOWS.md."
    }
}
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE." }
}
function Get-SmiPath {
    if ($NvidiaSmi) {
        if (-not (Test-Path -LiteralPath $NvidiaSmi -PathType Leaf)) {
            throw "Specified nvidia-smi executable does not exist: $NvidiaSmi"
        }
        return (Resolve-Path -LiteralPath $NvidiaSmi).Path
    }
    $command = Get-Command nvidia-smi.exe,nvidia-smi -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command -and (Test-Path -LiteralPath $command.Source -PathType Leaf)) {
        return $command.Source
    }
    $candidates = @(
        (Join-Path $env:WINDIR 'System32\nvidia-smi.exe'),
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    # Newer NVIDIA display drivers may keep nvidia-smi inside the DriverStore
    # rather than the old System32 or NVSMI paths. This differs by installation.
    $driverStore = Join-Path $env:WINDIR 'System32\DriverStore\FileRepository'
    if (Test-Path -LiteralPath $driverStore -PathType Container) {
        $driverFolders = Get-ChildItem -LiteralPath $driverStore -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^(nv|nvidia)' }
        foreach ($folder in $driverFolders) {
            $found = Get-ChildItem -LiteralPath $folder.FullName -Filter 'nvidia-smi.exe' -File -Recurse -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($found) { return $found.FullName }
        }
    }
    throw 'nvidia-smi.exe not found in this PowerShell environment; CUDA may still work. To use telemetry, pass -NvidiaSmi with its full Windows path.'
}
function Get-ServerPath {
    $paths = @(
        (Join-Path $BuildDir 'bin\Release\llama-server.exe'),
        (Join-Path $BuildDir 'bin\llama-server.exe')
    )
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) { return $path }
    }
    throw "llama-server.exe not found in $BuildDir. Run -Action Build first."
}
function Test-Prerequisites {
    foreach ($name in @('git','cmake','py','nvcc')) { Require-Command $name }
    Write-Host 'Windows NVIDIA GPUs:'
    try {
        $smi = Get-SmiPath
        Invoke-Checked -Executable $smi -Arguments @('-L')
    } catch {
        Write-Warning "GPU telemetry unavailable in this shell: $($_.Exception.Message)"
    }
    Write-Host 'CUDA compiler:'
    Invoke-Checked -Executable 'nvcc' -Arguments @('--version')
    Write-Host 'Python:'
    Invoke-Checked -Executable 'py' -Arguments @('-3','--version')
    Write-Host 'CMake:'
    Invoke-Checked -Executable 'cmake' -Arguments @('--version')
    Write-Host 'Prerequisites found. Check llama-server --list-devices after building.'
}
function Invoke-Phase1 {
    param([ValidateSet('Smoke','Full','Profile')][string]$Kind)
    Require-Command 'py'
    $server = Get-ServerPath
    $smi = 'disabled'
    try {
        $smi = Get-SmiPath
        Write-Host "GPU telemetry: $smi"
    } catch {
        Write-Warning "nvidia-smi not found; continuing inference without GPU utilization telemetry. $($_.Exception.Message)"
    }
    $target = Join-Path $ModelDir 'Qwen3.8-27B-Q4_K_M.gguf'
    $mtp = Join-Path $ModelDir 'mtp-Qwen3.8-27B-Q4_0.gguf'
    foreach ($path in @($target,$mtp)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing $path. Run -Action Download."
        }
    }
    Invoke-Checked -Executable 'py' -Arguments @('-3',(Join-Path $toolDir 'source_audit.py'),$repoRoot)
    Write-Host 'llama.cpp CUDA device names (not necessarily nvidia-smi index order):'
    Invoke-Checked -Executable $server -Arguments @('--list-devices')
    if ($PrimaryDevice -eq $DraftDevice) {
        throw 'PrimaryDevice and DraftDevice must differ for the A/B/C comparison.'
    }
    $benchmarkArgs = @(
        '-3', (Join-Path $toolDir 'benchmark.py'),
        '--server', $server, '--target', $target, '--mtp', $mtp,
        '--llama-repo', $repoRoot, '--nvidia-smi', $smi,
        '--primary-device', $PrimaryDevice, '--draft-device', $DraftDevice,
        '--expect-primary', $ExpectPrimary, '--expect-draft', $ExpectDraft,
        '--target-ngl', [string]$TargetNgl
    )
    switch ($Kind) {
        'Smoke' {
            $benchmarkArgs += @('--contexts','4096','--depths','1,3',
                       '--generations','128','--prompt-profiles','short','--repeats','1',
                       '--output-dir',(Join-Path $toolDir 'results_windows_smoke'))
        }
        'Full' {
            $benchmarkArgs += @('--contexts','4096,16384','--depths','1,2,3,4',
                       '--generations','128,512','--prompt-profiles','short,long',
                       '--repeats','3','--output-dir',(Join-Path $toolDir 'results_windows_full'))
        }
        'Profile' {
            Require-Command 'nsys'
            $benchmarkArgs += @('--contexts','4096','--depths','3',
                       '--generations','256','--prompt-profiles','short',
                       '--repeats','1','--nsys','--trace',
                       '--extra-server-arg=--verbosity','--extra-server-arg=4',
                       '--output-dir',(Join-Path $toolDir 'results_windows_profile'))
        }
    }
    Invoke-Checked -Executable 'py' -Arguments $benchmarkArgs
}
switch ($Action) {
    'Check' {
        Test-Prerequisites
        $serverPresent = $false
        try { $server = Get-ServerPath; $serverPresent = $true } catch {}
        if ($serverPresent) { Invoke-Checked -Executable $server -Arguments @('--list-devices') }
    }
    'Build' {
        Require-Command 'cmake'
        Require-Command 'nvcc'
        Invoke-Checked -Executable 'cmake' -Arguments @(
            '-S',$repoRoot,'-B',$BuildDir,'-G','Visual Studio 17 2022','-A','x64',
            '-DGGML_CUDA=ON','-DLLAMA_BUILD_SERVER=ON','-DCMAKE_CUDA_ARCHITECTURES=86;120'
        )
        Invoke-Checked -Executable 'cmake' -Arguments @(
            '--build',$BuildDir,'--config','Release','--target','llama-server','--parallel','4'
        )
        $server = Get-ServerPath
        Invoke-Checked -Executable $server -Arguments @('--list-devices')
    }
    'Download' {
        Require-Command 'py'
        Invoke-Checked -Executable 'py' -Arguments @(
            '-3',(Join-Path $toolDir 'download_models_windows.py'),'--model-dir',$ModelDir
        )
        Invoke-Checked -Executable 'py' -Arguments @(
            '-3',(Join-Path $toolDir 'inspect_gguf.py'),
            (Join-Path $ModelDir 'Qwen3.8-27B-Q4_K_M.gguf'),
            (Join-Path $ModelDir 'mtp-Qwen3.8-27B-Q4_0.gguf')
        )
    }
    'Test' {
        Require-Command 'py'
        Invoke-Checked -Executable 'py' -Arguments @(
            '-3','-m','unittest','discover','-s',$toolDir,'-p','test_phase1.py','-v'
        )
    }
    'Smoke' { Invoke-Phase1 -Kind 'Smoke' }
    'Full' { Invoke-Phase1 -Kind 'Full' }
    'Profile' { Invoke-Phase1 -Kind 'Profile' }
    'Update' {
        Require-Command 'git'
        Require-Command 'cmake'
        Invoke-Checked -Executable 'git' -Arguments @(
            '-C',$repoRoot,'pull','--ff-only','origin','master'
        )
        Invoke-Checked -Executable 'cmake' -Arguments @(
            '--build',$BuildDir,'--config','Release','--target','llama-server','--parallel','4'
        )
        Write-Host 'Updated and rebuilt. Run -Action Test and -Action Smoke.'
    }
}
