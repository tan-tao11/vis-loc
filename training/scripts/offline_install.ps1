param(
    [Parameter(Mandatory=$true)][string]$BundleRoot,
    [Parameter(Mandatory=$true)][string]$Destination,
    [ValidateSet('Auto','A','B')][string]$Method = 'Auto',
    [string]$CondaExe = '',
    [string]$MinicondaDir = '',
    [switch]$InstallVCRuntime
)
. "$PSScriptRoot\common.ps1"
Test-ArtifactHashes $BundleRoot
$bundle = (Resolve-Path -LiteralPath $BundleRoot).Path
if ($Method -eq 'Auto') {
    if (Test-Path -LiteralPath (Join-Path $bundle 'environment_win64_py310_cu118.zip')) { $Method='A' }
    else { $Method='B' }
}
if ($Method -eq 'A' -and -not (Test-Path -LiteralPath (Join-Path $bundle 'environment_win64_py310_cu118.zip'))) {
    throw 'This standalone bundle uses Method B; the separate A environment archive is not present.'
}
Write-Host "Restoring integrated deployment using method $Method. No addon installation is required."
$dest = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $dest) {
    if (Get-ChildItem -LiteralPath $dest -Force | Select-Object -First 1) { throw 'Destination must be new/empty; do not overwrite existing environments.' }
}
New-Item -ItemType Directory -Path $dest -Force | Out-Null
if ($InstallVCRuntime) {
    $vc = Join-Path $bundle 'installers\vc_redist.x64.exe'
    if ((Get-AuthenticodeSignature -LiteralPath $vc).Status -ne 'Valid') { throw 'VC++ installer signature invalid' }
    $proc = Start-Process -FilePath $vc -ArgumentList '/install','/quiet','/norestart' -WindowStyle Hidden -PassThru -Wait
    if ($proc.ExitCode -notin @(0,1638,3010)) { throw "VC++ installer failed: $($proc.ExitCode)" }
}
$envDir = Join-Path $dest 'env'
if ($Method -eq 'A') {
    Expand-Archive -LiteralPath (Join-Path $bundle 'environment_win64_py310_cu118.zip') -DestinationPath $envDir
    Set-EnvironmentPath $envDir
    # Windows conda-pack creates Scripts\conda-unpack.exe, not bin/conda-unpack.
    $unpack = Join-Path $envDir 'Scripts\conda-unpack.exe'
    if (-not (Test-Path -LiteralPath $unpack)) { throw 'Missing Windows conda-unpack.exe' }
    Invoke-Checked $unpack @()
} else {
    if (-not $CondaExe) {
        if (-not $MinicondaDir) { $MinicondaDir = Join-Path $dest 'miniconda' }
        if (Test-Path -LiteralPath $MinicondaDir) { throw 'Choose a new MinicondaDir' }
        $installer = Join-Path $bundle 'installers\Miniconda3-py310_25.1.1-2-Windows-x86_64.exe'
        $installArgs = @('/InstallationType=JustMe','/RegisterPython=0','/AddToPath=0','/S',"/D=$MinicondaDir")
        $proc = Start-Process -FilePath $installer -ArgumentList $installArgs -WindowStyle Hidden -PassThru -Wait
        if ($proc.ExitCode -ne 0) { throw "Miniconda installation failed: $($proc.ExitCode)" }
        $CondaExe = Join-Path $MinicondaDir 'Scripts\conda.exe'
        if (-not (Test-Path -LiteralPath $CondaExe)) {
            # Some Windows hosts leave only the installer's standalone bootstrap.
            # It is an offline-capable conda executable; verify it before use.
            $CondaExe = Join-Path $MinicondaDir '_conda.exe'
            if (-not (Test-Path -LiteralPath $CondaExe)) { throw 'Neither installed conda nor standalone bootstrap exists' }
            Write-Host 'Using installer-provided standalone conda bootstrap for explicit offline reconstruction.'
            Invoke-Checked $CondaExe @('--version')
        }
    }
    $spec = @('@EXPLICIT')
    foreach ($pkg in (Get-Content -LiteralPath (Join-Path $bundle 'conda_explicit.json') -Raw -Encoding UTF8 | ConvertFrom-Json)) {
        $pkgFile = Join-Path $bundle ('conda_packages\' + $pkg.filename)
        $spec += ([Uri]$pkgFile).AbsoluteUri
    }
    $specFile = Join-Path $dest 'conda-local-explicit.txt'
    [IO.File]::WriteAllLines($specFile, $spec, [Text.UTF8Encoding]::new($false))
    $env:CONDA_OFFLINE = 'true'; $env:CONDA_NO_PLUGINS = 'true'; $env:CONDA_SOLVER = 'classic'
    Invoke-Checked $CondaExe @('create','--offline','--no-default-packages','--prefix',$envDir,'--file',$specFile,'-y')
    Set-EnvironmentPath $envDir
    Invoke-Checked (Join-Path $envDir 'python.exe') @('-m','pip','install','--no-index','--require-hashes','--find-links',(Join-Path $bundle 'wheelhouse'),'-r',(Join-Path $bundle 'requirements.lock.txt'))
}
$project = Join-Path $dest 'project'
Expand-Archive -LiteralPath (Join-Path $bundle 'source.zip') -DestinationPath $project
Expand-Archive -LiteralPath (Join-Path $bundle 'model_and_source.zip') -DestinationPath $project -Force
Invoke-Checked (Join-Path $envDir 'python.exe') @('-m','pip','check')
Write-Host 'DINO weights are not bundled. Before offline_check or training, set model.checkpoint and model.checkpoint_sha256 in project\configs\train.yaml to the existing server DINO checkpoint. See README.'
Write-Host 'Included: MLP retrieval head for slant (oblique) IR, geometry and visualization tools.'
Write-Host "Restored to $dest. Run project\scripts\offline_check.ps1 with -EnvDir $envDir -BundleRoot $bundle. Do not move this environment after conda-unpack."
