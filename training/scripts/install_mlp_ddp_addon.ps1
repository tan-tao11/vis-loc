param([Parameter(Mandatory=$true)][string]$ProjectDir,[string]$AddonRoot='')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (-not $AddonRoot) { $AddonRoot=$PSScriptRoot }
$projectPath=(Resolve-Path -LiteralPath $ProjectDir).Path
$addonPath=(Resolve-Path -LiteralPath $AddonRoot).Path
function Read-Checked {
    param([string]$Root,[string]$Manifest)
    $result=@()
    foreach ($line in Get-Content -LiteralPath (Join-Path $Root $Manifest) -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts=$line -split '\s+',2
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[a-fA-F0-9]{64}$') { throw 'Invalid addon checksum manifest' }
        $target=[IO.Path]::GetFullPath((Join-Path $Root $parts[1]))
        if (-not $target.StartsWith($Root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe addon path' }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $parts[0].ToLower()) { throw "Checksum mismatch: $target" }
        $result += [PSCustomObject]@{Relative=$parts[1];Hash=$parts[0]}
    }
    return $result
}
$entries=@(Read-Checked $addonPath 'MLP_DDP_ADDON_SHA256SUMS.txt')
# Base code validation reads the user's project, but never compares or changes data configs.
foreach ($line in Get-Content -LiteralPath (Join-Path $addonPath 'MLP_DDP_BASE_SHA256SUMS.txt') -Encoding UTF8) {
    $parts=$line -split '\s+',2
    $target=[IO.Path]::GetFullPath((Join-Path $projectPath $parts[1]))
    if (-not $target.StartsWith($projectPath.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe base path' }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $parts[0].ToLower()) { throw "Base version mismatch: $target" }
}
$targets=@()
foreach ($entry in $entries) {
    if (-not $entry.Relative.StartsWith('payload/')) { continue }
    $relative=$entry.Relative.Substring('payload/'.Length)
    $destination=[IO.Path]::GetFullPath((Join-Path $projectPath $relative))
    if (-not $destination.StartsWith($projectPath.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe destination' }
    # Additive-only: every conflict is rejected BEFORE writing any file.
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $entry.Hash.ToLower()) {
            throw "Existing file differs; addon refuses to overwrite: $destination"
        }
    } else { $targets += [PSCustomObject]@{Source=(Join-Path $addonPath $entry.Relative);Destination=$destination} }
}
foreach ($target in $targets) {
    $null=New-Item -ItemType Directory -Path (Split-Path $target.Destination -Parent) -Force
    Copy-Item -LiteralPath $target.Source -Destination $target.Destination
}
Write-Host "MLP DDP addon installed: $($targets.Count) new files. All existing files preserved."
Write-Host 'Run scripts/train_mlp_ddp.ps1 -EnvDir <environment path> to train on two distinct GPUs.'
