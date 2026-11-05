param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Run='artifacts/real', [ValidateSet('val','test')][string]$Split='val')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
Invoke-Checked (Join-Path $EnvDir 'python.exe') @('evaluate.py','--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--checkpoint',"$Run/training/best.pth",'--split',$Split,'--output',"$Run/evaluation_$Split.json")
# evaluate.py includes per-method contact sheets in a unique report directory.
