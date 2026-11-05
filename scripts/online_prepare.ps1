param(
    [string]$EnvDir = 'D:\IR_Retrieval_build\env',
    [string]$BundleRoot = 'D:\IR_Retrieval_delivery',
    [string]$Conda = 'conda',
    [string]$TargetNvidiaSmi = ''
)
. "$PSScriptRoot\common.ps1"
$project = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $project
if ($TargetNvidiaSmi) { Get-Content -LiteralPath $TargetNvidiaSmi } else { Write-Warning 'Target driver unknown: cu118 remains a candidate; validate on target before acceptance.' }
# 不修改宿主机驱动，不更新已有环境。官方 Windows CUDA11.8 工具链配套驱动基线 522.06。
if (-not (Test-Path -LiteralPath (Join-Path $EnvDir 'python.exe'))) {
    Invoke-Checked $Conda @('create','--prefix',$EnvDir,'--no-default-packages','--override-channels','-c','https://repo.anaconda.com/pkgs/main','python=3.10.16','pip=25.0','setuptools=75.8.0','-y')
}
Set-EnvironmentPath $EnvDir
$python = Join-Path $EnvDir 'python.exe'
$wheels = Join-Path $BundleRoot 'wheelhouse'
New-Item -ItemType Directory -Path $wheels -Force | Out-Null
Invoke-Checked $python @('-m','pip','download','--only-binary=:all:','--dest',$wheels,'-r','requirements.in','--index-url','https://pypi.org/simple')
Invoke-Checked $python @('-m','pip','download','--only-binary=:all:','--dest',$wheels,'torch==2.5.1+cu118','torchvision==0.20.1+cu118','--index-url','https://download.pytorch.org/whl/cu118')
Invoke-Checked $python @('-m','pip','install','--no-index','--find-links',$wheels,'-r','requirements.in','torch==2.5.1+cu118','torchvision==0.20.1+cu118')
Invoke-Checked $python @('scripts/download_assets.py','--bundle',$BundleRoot)
Invoke-Checked $python @('-m','pip','check')
Write-Host 'Dependencies ready. Run tests and offline_check, then pack_environment.ps1.'
