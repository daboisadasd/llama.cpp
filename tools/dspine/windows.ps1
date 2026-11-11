# Native Windows checkpoint preflight for Tencent Qwen3.8-27B DSpine.
# This does not launch DSpine inference. We refuse silent fallback to DFlash/MTP.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Check','Prepare','Convert','Test','GraphTest','SourceAudit')]
    [string]$Action,
    [string]$CheckpointDir = (Join-Path (Join-Path $env:USERPROFILE 'models') 'Qwen3.8-27B-DSpine-b16'),
    [string]$ModelDir = (Join-Path (Join-Path $env:USERPROFILE 'models') 'Qwen3.8-27B')
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Here = Split-Path -Parent $PSCommandPath
$Root = (Resolve-Path (Join-Path $Here '..\..')).Path
$Python = 'py'
$Bf16Head = Join-Path $ModelDir 'mtp-Qwen3.8-27B-BF16.gguf'
$ReportDir = Join-Path $Here 'results_preflight'

function Invoke-Checked {
    param([string[]]$CommandLine)
    & $Python @CommandLine
    if ($LASTEXITCODE -ne 0) { throw "Python returned exit code $LASTEXITCODE" }
}

switch ($Action) {
    'Test' {
        Invoke-Checked -CommandLine @('-3','-m','unittest','discover','-s',$Here,'-p','test_*.py','-v')
    }
    'Convert' {
        # Experimental tensor-package only; runtime llama.cpp cannot load general.architecture=dspine yet.
        New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
        $output = Join-Path $CheckpointDir 'Qwen3.8-27B-DSpine-b16.gguf'
        if (-not (Test-Path -LiteralPath $Bf16Head -PathType Leaf)) {
            throw "Missing BF16 target output head: $Bf16Head. PrepareMtp BF16 via tools/remote_mtp/windows.ps1 first."
        }
        Invoke-Checked -CommandLine @('-3',(Join-Path $Here 'convert_gguf.py'),
            '--model-dir',$CheckpointDir,'--target-bf16-gguf',$Bf16Head,
            '--output',$output,'--report',(Join-Path $ReportDir 'conversion_report.json'))
        Write-Host 'DSpine GGUF tensor package created; native runtime is still under development.'
    }
    'GraphTest' {
        $dir = Join-Path $Root 'build-dspine-bridge-win'
        & cmake -S $Here -B $dir -G 'Visual Studio 17 2022' -A x64 '-DBUILD_SHARED_LIBS=OFF'
        if ($LASTEXITCODE -ne 0) { throw 'DSpine GGML CPU graph CMake configuration failed' }
        & cmake --build $dir --config Release --target test-dspine-bridge --parallel 4
        if ($LASTEXITCODE -ne 0) { throw 'DSpine GGML CPU graph compilation failed' }
        & ctest --test-dir $dir -C Release --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw 'DSpine GGML CPU graph numerical test failed' }
        Write-Host 'Native GGML code bridge CPU graph test passed (not yet full DSpine inference).'
    }
    'SourceAudit' {
        Invoke-Checked -CommandLine @('-3',(Join-Path $Here 'source_audit.py'),'--llama-root',$Root)
    }
    'Check' {
        # Download only the tiny config.json; no 4GB weight download yet.
        Invoke-Checked -CommandLine @('-3',(Join-Path $Here 'checkpoint.py'),'--model-dir',$CheckpointDir,'--download','--config-only')
    }
    'Prepare' {
        New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
        $args = @('-3',(Join-Path $Here 'checkpoint.py'),'--model-dir',$CheckpointDir,
                  '--download','--output',(Join-Path $ReportDir 'checkpoint_report.json'))
        if (Test-Path -LiteralPath $Bf16Head -PathType Leaf) {
            Write-Host "Checking candidate BF16 target projection from: $Bf16Head"
            $args += @('--bf16-mtp-gguf',$Bf16Head)
        } else {
            Write-Warning "BF16 MTP GGUF not available: $Bf16Head. This is optional for checkpoint validation."
        }
        Invoke-Checked -CommandLine $args
        Write-Host 'Checkpoint validated. DSpine llama.cpp execution is NOT yet implemented.'
    }
}
