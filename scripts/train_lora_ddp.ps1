param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/lora_ddp.yaml',
      [string]$Manifest='artifacts/real/manifest.json', [string]$Run='artifacts/real/lora_ddp',
      [string]$Devices='cuda:0,cuda:1', [string]$Resume='')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$argsList=@('train_lora_ddp.py','--config',$Config,'--manifest',$Manifest,'--output',$Run,'--devices',$Devices)
if ($Resume) { $argsList+=@('--resume',$Resume) }
Invoke-Checked (Join-Path $EnvDir 'python.exe') $argsList
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('export_lora.py','--checkpoint',"$Run/best.pth",'--output',"$Run/scc_retrieval_lora.pth")
