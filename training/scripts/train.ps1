param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/train.yaml', [string]$Run='artifacts/real', [string]$Device='cuda:0')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('train.py','--config',$Config,'--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--output',"$Run/training",'--device',$Device)
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('export.py','--checkpoint',"$Run/training/best.pth",'--output',"$Run/scc_retrieval_head.pth")
