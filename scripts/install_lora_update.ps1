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
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[a-fA-F0-9]{64}$') { throw 'Invalid update checksum manifest' }
        $target = [IO.Path]::GetFullPath((Join-Path $Root $parts[1]))
        if (-not $target.StartsWith($Root.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe update path' }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $parts[0].ToLower()) {
            throw "Version or checksum mismatch: $target"
        }
        $entries += $parts[1]
    }
    return $entries
}
# Verify the complete package and required original code BEFORE writing any file.
$files = @(Read-VerifiedEntries $updatePath (Join-Path $updatePath 'LORA_UPDATE_SHA256SUMS.txt'))
$null = Read-VerifiedEntries $projectPath (Join-Path $updatePath 'LORA_BASE_SHA256SUMS.txt')
foreach ($relative in $files) {
    if (-not $relative.StartsWith('payload/')) { continue }
    $suffix = $relative.Substring('payload/'.Length)
    if ($suffix -eq 'configs/train.yaml') { throw 'Update must preserve the original data config' }
    $destination = [IO.Path]::GetFullPath((Join-Path $projectPath $suffix))
    if (-not $destination.StartsWith($projectPath.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe destination' }
    $null = New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force
    Copy-Item -LiteralPath (Join-Path $updatePath $relative) -Destination $destination -Force
}
Write-Host "LoRA update installed in $projectPath. Existing environment and data configuration retained."
