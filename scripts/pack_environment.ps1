param([string]$EnvDir = 'D:\IR_Retrieval_build\env', [string]$BundleRoot = 'D:\IR_Retrieval_delivery')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('scripts/package_delivery.py','--env',$EnvDir,'--bundle',$BundleRoot)
