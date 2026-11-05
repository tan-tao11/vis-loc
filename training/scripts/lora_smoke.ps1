param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Device='cuda:0', [string]$Output='artifacts/lora_smoke')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('lora_smoke.py','--device',$Device,'--output',$Output)
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('-m','pytest','-q')
