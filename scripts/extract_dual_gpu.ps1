param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config = 'configs/train.yaml',
      [string]$Manifest = 'artifacts/real/manifest.json', [string]$Cache = 'artifacts/real/features', [int]$GPUs = 2)
. "$PSScriptRoot\common.ps1"
if ($GPUs -notin @(1,2)) { throw 'Choose 1 or 2 independent GPU workers' }
Set-EnvironmentPath $EnvDir
$project = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $project
New-Item -ItemType Directory -Path $Cache -Force | Out-Null
$python = Join-Path $EnvDir 'python.exe'; $processes = @()
for ($i = 0; $i -lt $GPUs; $i++) {
    $arguments = @('extract_features.py','--config',('"'+$Config+'"'),'--manifest',('"'+$Manifest+'"'),'--cache',('"'+$Cache+'"'),'--device',"cuda:$i",'--shard-index',"$i",'--num-shards',"$GPUs")
    $processes += Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Cache "worker_$i.log") -RedirectStandardError (Join-Path $Cache "worker_$i.err")
}
foreach ($process in $processes) { $process.WaitForExit(); if ($process.ExitCode -ne 0) { throw "Extraction worker failed: $($process.ExitCode). Check worker logs; rerun to resume." } }
Invoke-Checked $python @('extract_features.py','--config',$Config,'--manifest',$Manifest,'--cache',$Cache,'--merge','--num-shards',"$GPUs")
