param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$DataRun='artifacts/real',
      [string]$Run='', [string]$Devices='cuda:0,cuda:1', [string]$Resume='')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
if (-not $Run) { $Run=Join-Path $DataRun 'mlp_ddp' }
$argsList=@('train_mlp_ddp.py','--manifest',"$DataRun/manifest.json",'--cache',"$DataRun/features",'--output',$Run,'--devices',$Devices)
if ($Resume) { $argsList+=@('--resume',$Resume) }
Invoke-Checked (Join-Path $EnvDir 'python.exe') $argsList
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('export.py','--checkpoint',"$Run/best.pth",'--output',"$Run/scc_retrieval_head.pth")
