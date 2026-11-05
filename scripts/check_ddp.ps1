param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Devices='cuda:0,cuda:1',
      [string]$Output='artifacts/ddp_check', [switch]$FullSmoke)
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$entry='ddp_check.py'
if ($FullSmoke) { $entry='ddp_smoke.py' }
Invoke-Checked (Join-Path $EnvDir 'python.exe') @($entry,'--devices',$Devices,'--output',$Output)
