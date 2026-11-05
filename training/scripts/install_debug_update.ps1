param([Parameter(Mandatory=$true)][string]$ProjectDir, [string]$UpdateRoot='')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $UpdateRoot) { $UpdateRoot = $PSScriptRoot }
$projectPath = (Resolve-Path -LiteralPath $ProjectDir).Path
$updatePath = (Resolve-Path -LiteralPath $UpdateRoot).Path
function Read-VerifiedEntries {
    param([string]$Root, [string]$ManifestFile)
    $entries = @()
    foreach ($line in Get-Content -LiteralPath $ManifestFile -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\s+', 2
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[a-fA-F0-9]{64}$') { throw 'Invalid checksum manifest' }
        $target = [IO.Path]::GetFullPath((Join-Path $Root $parts[1]))
        if (-not $target.StartsWith($Root.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe path' }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $parts[0].ToLower()) { throw "Version or checksum mismatch: $target" }
        $entries += $parts[1]
    }
    return $entries
}
$files = @(Read-VerifiedEntries $updatePath (Join-Path $updatePath 'DEBUG_UPDATE_SHA256SUMS.txt'))
$null = Read-VerifiedEntries $projectPath (Join-Path $updatePath 'DEBUG_BASE_SHA256SUMS.txt')
$backup = Join-Path $projectPath ('artifacts\code_backups\debug_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
# Validate every destination before copying anything.
$targets = @()
foreach ($relative in $files) {
    if (-not $relative.StartsWith('payload/')) { continue }
    $suffix = $relative.Substring('payload/'.Length)
    if ($suffix.StartsWith('configs/')) { throw 'Update must preserve all configurations' }
    $destination = [IO.Path]::GetFullPath((Join-Path $projectPath $suffix))
    if (-not $destination.StartsWith($projectPath.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe destination' }
    $targets += [PSCustomObject]@{Relative=$relative; Suffix=$suffix; Destination=$destination}
}
foreach ($item in $targets) {
    if (Test-Path -LiteralPath $item.Destination) {
        $saved = Join-Path $backup $item.Suffix
        $null = New-Item -ItemType Directory -Path (Split-Path $saved -Parent) -Force
        Copy-Item -LiteralPath $item.Destination -Destination $saved
    }
}
foreach ($item in $targets) {
    $null = New-Item -ItemType Directory -Path (Split-Path $item.Destination -Parent) -Force
    Copy-Item -LiteralPath (Join-Path $updatePath $item.Relative) -Destination $item.Destination -Force
}
Write-Host "Debug/output update installed: $projectPath"
Write-Host "Previous code backed up: $backup"
Write-Host 'Configurations, DINO weights and the existing environment were preserved. Start new training runs after a code update.'
