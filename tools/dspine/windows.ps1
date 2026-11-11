# Native Windows checkpoint preflight for Tencent Qwen3.8-27B DSpine.
# This does not launch DSpine inference. We refuse silent fallback to DFlash/MTP.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Check','Prepare','Test','SourceAudit')]
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
