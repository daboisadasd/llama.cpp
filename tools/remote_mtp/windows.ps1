# Native Windows host for remote MTP using Linux GGML RPC, no WSL and no eGPU required.
# Run in "Developer PowerShell for VS 2022".
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Build','Check','Smoke','Repeat','Full','Trace','Test','Update')]
    [string]$Action,
    [string]$Worker = '',
    [string]$PrimaryDevice = 'CUDA0',
    [string]$RpcDraftDevice = 'RPC0',
    [string]$ExpectPrimary = '5080',
    [ValidateRange(0,256)] [int]$TargetNgl = 35,
    [ValidateSet('on','off','auto')] [string]$FlashAttn = 'on',
    [string]$ModelDir = (Join-Path (Join-Path $env:USERPROFILE 'models') 'Qwen3.8-27B'),
    [string]$BuildDir = '',
    [string]$CudaArchitectures = '120'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ToolDir = Split-Path -Parent $PSCommandPath
$Root = (Resolve-Path (Join-Path $ToolDir '..\..')).Path
if (-not $BuildDir) { $BuildDir = Join-Path $Root 'build-win' }
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable exited with code $LASTEXITCODE" }
}
function Get-Server {
    $paths = @(
        (Join-Path $BuildDir 'bin\Release\llama-server.exe'),
        (Join-Path $BuildDir 'bin\llama-server.exe')
    )
    foreach ($candidate in $paths) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw 'Missing Windows llama-server.exe. Run -Action Build first.'
}
function Invoke-RpcTest {
    param([string]$Mode)
    if (-not $Worker) {
        throw 'Provide -Worker PRIVATE_LAN_IP:50052. Start worker_linux.sh serve on Linux first.'
    }
    if ($Worker -notmatch '^[^:]+:[0-9]+$') {
        throw 'Expected -Worker PRIVATE_IP:PORT, e.g. 192.168.1.80:50052.'
    }
    if ($PrimaryDevice -eq $RpcDraftDevice) { throw 'Target and drafting devices must differ.' }
    $server = Get-Server
    $target = Join-Path $ModelDir 'Qwen3.8-27B-Q4_K_M.gguf'
    $mtp = Join-Path $ModelDir 'mtp-Qwen3.8-27B-Q4_0.gguf'
    $cmd = @(
        '-3',(Join-Path $ToolDir 'rpc_benchmark.py'),
        '--action',$Mode,'--server',$server,'--worker',$Worker,
        '--primary-device',$PrimaryDevice,'--rpc-draft-device',$RpcDraftDevice,
        '--expect-primary',$ExpectPrimary,
        '--target',$target,'--mtp',$mtp,
        '--target-ngl',[string]$TargetNgl,'--flash-attn',$FlashAttn,
        '--llama-repo',$Root,
        '--output-dir',(Join-Path $ToolDir ('results_' + $Mode + '_fa_' + $FlashAttn))
    )
    Invoke-Checked -Executable 'py' -Arguments $cmd
}
switch ($Action) {
    'Build' {
        # Enable RPC in the existing Windows build; model target remains explicitly CUDA0.
        Invoke-Checked -Executable 'cmake' -Arguments @(
            '-S',$Root,'-B',$BuildDir,
            '-G','Visual Studio 17 2022','-A','x64',
            '-DGGML_CUDA=ON','-DGGML_RPC=ON','-DLLAMA_BUILD_SERVER=ON',
            ('-DCMAKE_CUDA_ARCHITECTURES=' + $CudaArchitectures)
        )
        Invoke-Checked -Executable 'cmake' -Arguments @(
            '--build',$BuildDir,'--config','Release','--target','llama-server','--parallel','4'
        )
        Invoke-Checked -Executable (Get-Server) -Arguments @('--list-devices')
    }
    'Test' {
        Invoke-Checked -Executable 'py' -Arguments @(
            '-3','-m','unittest','discover','-s',$ToolDir,'-p','test_*.py','-v'
        )
    }
    'Check' { Invoke-RpcTest 'check' }
    'Smoke' { Invoke-RpcTest 'smoke' }
    'Repeat' { Invoke-RpcTest 'repeat' }
    'Full' { Invoke-RpcTest 'full' }
    'Trace' { Invoke-RpcTest 'trace' }
    'Update' {
        Invoke-Checked -Executable 'git' -Arguments @('-C',$Root,'pull','--ff-only','origin','master')
        Invoke-Checked -Executable 'cmake' -Arguments @(
            '--build',$BuildDir,'--config','Release','--target','llama-server','--parallel','4'
        )
        Write-Host 'Updated Windows RPC host. Rebuild/sync the Linux worker separately.'
    }
}
