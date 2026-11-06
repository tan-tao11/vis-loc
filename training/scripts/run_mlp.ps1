# One-shot MLP pipeline: prepare -> dual-GPU feature extraction -> train/export -> val evaluation. Output dir = config "output".
# Use -Limit 12 for a small check first; -Steps selects stages, e.g. -Steps prepare  or  -Steps extract,train,eval
# GPUs: only CUDA devices with >= 6 GB are used (two A4000; a small display GPU such as a P400 is skipped).
# -DeviceIds '0,1' overrides the automatic choice (torch CUDA numbering, see the 'Using GPUs' line).
param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/slant.yaml', [int]$GPUs=2,
      [int]$Limit=0, [string[]]$Steps=@('prepare,extract,train,eval'), [string[]]$DeviceIds=@())
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $EnvDir 'python.exe'
# '-Steps a,b' arrives as an array in a PowerShell console and as one string via powershell -File: accept both.
$StepList = @(($Steps -join ',') -split '[,\s]+' | Where-Object { $_ })
$bad = @($StepList | Where-Object { $_ -notin @('prepare','extract','train','eval') })
if ($bad.Count) { throw "Unknown step(s): $($bad -join ', '). Use prepare,extract,train,eval" }
$Run = (& $python -c "import sys; from common import config; print(config(sys.argv[1])['output'])" $Config).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Run) { throw "Cannot read output from $Config" }
$Ids = @()
if (@($StepList | Where-Object { $_ -in @('extract','train','eval') }).Count) {
    if ($DeviceIds.Count) { $Ids = @(($DeviceIds -join ',') -split '[,\s]+' | Where-Object { $_ }) }
    else {
        $found = (& $python -c "from common import usable_cuda_devices; print(','.join(map(str, usable_cuda_devices())))").Trim()
        if ($LASTEXITCODE -ne 0) { throw 'GPU query failed (is CUDA PyTorch installed in the environment?)' }
        $Ids = @($found -split ',' | Where-Object { $_ })
    }
    if (-not $Ids.Count) { throw 'No CUDA GPU with >= 6 GB memory found' }
    if ($Ids.Count -lt $GPUs) { $GPUs = $Ids.Count }
    $Ids = @($Ids[0..($GPUs-1)])
    & $python -c "import sys, torch; [print('Using GPU cuda:%s = %s (%.1f GB)' % (i, torch.cuda.get_device_name(int(i)), torch.cuda.get_device_properties(int(i)).total_memory/2**30)) for i in sys.argv[1:]]" @Ids
}
if ($StepList -contains 'prepare') {
    $a = @('prepare_dataset.py','--config',$Config); if ($Limit -gt 0) { $a += @('--limit',"$Limit") }
    Invoke-Checked $python $a
    # Visual check only: a timestamped directory (reruns never collide) and failures only warn.
    $diag = "$Run/diagnostics_prepare_" + (Get-Date -Format 'yyyyMMdd_HHmmss')
    & $python diagnostics.py --config $Config --manifest "$Run/manifest.json" --output $diag
    if ($LASTEXITCODE -ne 0) { Write-Warning "Contact sheet incomplete (exit $LASTEXITCODE); see $diag/contact_sheet.png.json" }
}
if ($StepList -contains 'extract') {
    & "$PSScriptRoot\extract_dual_gpu.ps1" -EnvDir $EnvDir -Config $Config -Manifest "$Run/manifest.json" -Cache "$Run/features" -GPUs $GPUs -DeviceIds ($Ids -join ',')
}
if ($StepList -contains 'train') {
    Invoke-Checked $python @('train.py','--config',$Config,'--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--output',"$Run/training",'--device',"cuda:$($Ids[0])")
    Invoke-Checked $python @('export.py','--checkpoint',"$Run/training/best.pth",'--output',"$Run/scc_retrieval_head.pth")
}
if ($StepList -contains 'eval') {
    Invoke-Checked $python @('evaluate.py','--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--checkpoint',"$Run/training/best.pth",'--split','val','--output',"$Run/evaluation_val.json",'--device',"cuda:$($Ids[0])")
}
