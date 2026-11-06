param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$DataRun='artifacts/real',
      [string]$Run='', [string]$Devices='', [string]$Resume='')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
if (-not $Run) { $Run=Join-Path $DataRun 'mlp_ddp' }
if (-not $Devices) {   # first two CUDA GPUs with >= 6 GB (torch numbering); a small display GPU is never used
    $Devices = (& (Join-Path $EnvDir 'python.exe') -c "from common import usable_cuda_devices as u; d=u(); assert len(d)>=2, 'need two GPUs with >= 6 GB'; print(','.join('cuda:%d' % i for i in d[:2]))").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Devices) { throw 'Could not find two CUDA GPUs with >= 6 GB' }
}
Write-Host "DDP devices: $Devices"
$argsList=@('train_mlp_ddp.py','--manifest',"$DataRun/manifest.json",'--cache',"$DataRun/features",'--output',$Run,'--devices',$Devices)
if ($Resume) { $argsList+=@('--resume',$Resume) }
Invoke-Checked (Join-Path $EnvDir 'python.exe') $argsList
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('export.py','--checkpoint',"$Run/best.pth",'--output',"$Run/scc_retrieval_head.pth")
