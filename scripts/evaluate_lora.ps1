param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Manifest='artifacts/real/manifest.json',
      [string]$Run='artifacts/real/lora', [ValidateSet('val','test')][string]$Split='val',
      [string]$Device='cuda:0', [string]$MlpCheckpoint='')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$argsList = @('evaluate_lora.py','--manifest',$Manifest,'--checkpoint',"$Run/best.pth",'--split',$Split,'--device',$Device,'--output',"$Run/evaluation_$Split.json")
if ($MlpCheckpoint) { $argsList += @('--mlp-checkpoint',$MlpCheckpoint) }
Invoke-Checked (Join-Path $EnvDir 'python.exe') $argsList
