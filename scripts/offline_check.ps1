param([Parameter(Mandatory=$true)][string]$EnvDir, [Parameter(Mandatory=$true)][string]$BundleRoot,
      [int]$ExpectedGPUs = 2, [switch]$RequireA4000, [string]$Device = 'cuda:0')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
Test-ArtifactHashes $BundleRoot
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('-m','pip','check')
$argsList = @('offline_check.py','--expected-gpus',"$ExpectedGPUs",'--device',$Device,'--bundle-root',$BundleRoot,'--hashes',(Join-Path $BundleRoot 'SHA256SUMS.txt'))
if ($RequireA4000) { $argsList += '--require-a4000' }
Invoke-Checked (Join-Path $EnvDir 'python.exe') $argsList
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('-m','pytest','-q')
