# Native Windows DSpine development actions.
# The RPC bridge/tail actions are experimental GGML graph tests, not a
# complete DSpine speculative decoding engine.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Check','Prepare','Convert','Test','GraphTest','RpcBridgeTest','SourceAudit')]
    [string]$Action,
    [string]$CheckpointDir = (Join-Path (Join-Path $env:USERPROFILE 'models') 'Qwen3.8-27B-DSpine-b16'),
    [string]$Worker = '',
    [switch]$Tail,
    [switch]$RestartPartial,
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
        # Download config only; the checkpoint is already cached separately.
        Invoke-Checked -CommandLine @('-3',(Join-Path $Here 'checkpoint.py'),
            '--model-dir',$CheckpointDir,'--download','--config-only')
    }
    'Prepare' {
        New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
        $arguments = @('-3',(Join-Path $Here 'checkpoint.py'),'--model-dir',$CheckpointDir,
                       '--download','--output',(Join-Path $ReportDir 'checkpoint_report.json'))
        if (Test-Path -LiteralPath $Bf16Head -PathType Leaf) {
            Write-Host "Checking candidate BF16 target output head: $Bf16Head"
            $arguments += @('--bf16-mtp-gguf',$Bf16Head)
        } else {
            Write-Warning "Optional BF16 target head missing: $Bf16Head"
        }
        Invoke-Checked -CommandLine $arguments
        Write-Host 'DSpine checkpoint validated. The full native speculative engine is not yet available.'
    }
    'Convert' {
        New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
        $output = Join-Path $CheckpointDir 'Qwen3.8-27B-DSpine-b16.gguf'
        $partial = "$output.partial"
        if (Test-Path -LiteralPath $output -PathType Leaf) {
            throw "Output GGUF already exists: $output. No overwrite will be attempted."
        }
        if (Test-Path -LiteralPath $partial -PathType Leaf) {
            if ($RestartPartial) {
                # Only the converter's exact temporary output is removed,
                # never the downloaded safetensors or an existing GGUF.
                Write-Host "Removing interrupted conversion artifact: $partial"
                Remove-Item -LiteralPath $partial -Force
            } else {
                throw "Interrupted converter artifact exists: $partial. Rerun -Action Convert -RestartPartial to remove only this temporary file."
            }
        }
        if (-not (Test-Path -LiteralPath $Bf16Head -PathType Leaf)) {
            throw "Missing BF16 target head $Bf16Head. Prepare it via tools/remote_mtp/windows.ps1 -Action PrepareMtp -MtpPrecision BF16."
        }
        Invoke-Checked -CommandLine @('-3',(Join-Path $Here 'convert_gguf.py'),
            '--model-dir',$CheckpointDir,'--target-bf16-gguf',$Bf16Head,
            '--output',$output,'--report',(Join-Path $ReportDir 'conversion_report.json'))
        Write-Host 'DSpine GGUF generated as an experimental tensor package (not yet loadable by llama-server).'
    }
    'GraphTest' {
        $dir = Join-Path $Root 'build-dspine-bridge-win'
        & cmake -S $Here -B $dir -G 'Visual Studio 17 2022' -A x64 '-DBUILD_SHARED_LIBS=OFF'
        if ($LASTEXITCODE -ne 0) { throw 'DSpine GGML CPU graph configuration failed' }
        & cmake --build $dir --config Release --target test-dspine-bridge test-dspine-refine test-dspine-tail test-dspine-tables test-dspine-gguf-names test-dspine-bridge-bf16 --parallel 4
        if ($LASTEXITCODE -ne 0) { throw 'DSpine GGML CPU graph compilation failed' }
        & ctest --test-dir $dir -C Release --output-on-failure
        if ($LASTEXITCODE -ne 0) { throw 'DSpine GGML CPU graph numerical test failed' }
        Write-Host 'DSpine GGML CPU graph tests passed (not full inference).'
    }
    'RpcBridgeTest' {
        if ($Worker -notmatch '^[^:]+:[0-9]+$') {
            throw 'Pass -Worker PRIVATE_IP:PORT, e.g. 192.168.1.13:50052'
        }
        $model = Join-Path $CheckpointDir 'Qwen3.8-27B-DSpine-b16.gguf'
        if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
            throw "DSpine GGUF missing: $model. Run -Action Convert first."
        }
        $dir = Join-Path $Root 'build-dspine-rpc-win'
        & cmake -S $Here -B $dir -G 'Visual Studio 17 2022' -A x64 '-DBUILD_SHARED_LIBS=OFF' '-DDSPINE_BRIDGE_RPC=ON'
        if ($LASTEXITCODE -ne 0) { throw 'DSpine RPC bridge CMake configuration failed' }
        & cmake --build $dir --config Release --target test-dspine-rpc-bridge --parallel 4
        if ($LASTEXITCODE -ne 0) { throw 'DSpine RPC bridge compilation failed' }
        $exe = Join-Path $dir 'Release\test-dspine-rpc-bridge.exe'
        if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
            throw "DSpine RPC test executable missing: $exe"
        }
        $exeArgs = @($model, $Worker)
        if ($Tail) { $exeArgs += '--tail' }
        & $exe @exeArgs
        if ($LASTEXITCODE -ne 0) { throw 'Native DSpine RPC bridge graph execution failed' }
        if ($Tail) {
            Write-Host 'DSpine BF16 bridge, top-K and serial tail ran on RPC GPU. Full drafting/verification is pending.'
        } else {
            Write-Host 'DSpine BF16 bridge ran on RPC GPU. Full drafting/verification is pending.'
        }
    }
}
