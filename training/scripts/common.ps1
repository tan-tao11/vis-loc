$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}
function Set-EnvironmentPath {
    param([string]$EnvDir)
    $resolvedEnv = (Resolve-Path -LiteralPath $EnvDir).Path
    $env:PATH = "$resolvedEnv;$resolvedEnv\Scripts;$resolvedEnv\Library\bin;$env:PATH"
    $env:CONDA_PREFIX = $resolvedEnv
    $env:PROJ_NETWORK = 'OFF'
    $env:XFORMERS_DISABLED = '1'
}
function Test-ArtifactHashes {
    param([string]$BundleRoot, [string]$Manifest = 'SHA256SUMS.txt')
    $bundleResolved = (Resolve-Path -LiteralPath $BundleRoot).Path
    foreach ($line in Get-Content -LiteralPath (Join-Path $bundleResolved $Manifest) -Encoding UTF8) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $parts = $line -split '\s+', 2
        $target = [IO.Path]::GetFullPath((Join-Path $bundleResolved $parts[1]))
        if (-not $target.StartsWith($bundleResolved.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe checksum path' }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower() -ne $parts[0].ToLower()) { throw "Hash mismatch: $target" }
    }
}
