param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/train.yaml',
      [string]$Output='artifacts/geometry_validation', [int]$MaxFrames=16,
      [string]$Controls='', [double]$ToleranceM=0, [ValidateSet('gt','prior')][string]$PositionSource='gt')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$argsList = @('validate_geometry.py','--config',$Config,'--output',$Output,'--max-frames',"$MaxFrames",'--position-source',$PositionSource)
if ($Controls) { $argsList += @('--controls',$Controls) }
if ($ToleranceM -gt 0) { $argsList += @('--tolerance-m',$ToleranceM.ToString([Globalization.CultureInfo]::InvariantCulture)) }
& (Join-Path $EnvDir 'python.exe') @argsList
$diagnosticExit = $LASTEXITCODE
if ($diagnosticExit -eq 2) { Write-Host "Geometry needs attention; open $Output/index.html for missing inputs or errors. Calibration flags were not changed." }
elseif ($diagnosticExit -ne 0) { throw "Geometry validator failed: $diagnosticExit" }
exit $diagnosticExit
