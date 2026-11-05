param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/lora.yaml',
      [string]$Manifest='artifacts/real/manifest.json', [string]$Run='artifacts/real/lora',
      [string]$Device='cuda:0', [string]$Resume='')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$argsList = @('train_lora.py','--config',$Config,'--manifest',$Manifest,'--output',$Run,'--device',$Device)
if ($Resume) { $argsList += @('--resume',$Resume) }
Invoke-Checked (Join-Path $EnvDir 'python.exe') $argsList
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('export_lora.py','--checkpoint',"$Run/best.pth",'--output',"$Run/scc_retrieval_lora.pth")
