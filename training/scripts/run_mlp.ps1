# One-shot MLP pipeline: prepare -> dual-GPU feature extraction -> train/export -> val evaluation. Output dir = config "output".
# Use -Limit 12 for a small check first; -Steps selects stages, e.g. -Steps prepare  or  -Steps extract,train,eval
param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/mlp.yaml', [int]$GPUs=2,
      [int]$Limit=0, [string]$Steps='prepare,extract,train,eval')
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $EnvDir 'python.exe'
# powershell -File passes 'a,b' as one string; split it ourselves.
$StepList = @($Steps -split '[,\s]+' | Where-Object { $_ })
$bad = @($StepList | Where-Object { $_ -notin @('prepare','extract','train','eval') })
if ($bad.Count) { throw "Unknown step(s): $($bad -join ', '). Use prepare,extract,train,eval" }
$Run = (& $python -c "import sys; from common import config; print(config(sys.argv[1])['output'])" $Config).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Run) { throw "Cannot read output from $Config" }
if ($StepList -contains 'prepare') {
    $a = @('prepare_dataset.py','--config',$Config); if ($Limit -gt 0) { $a += @('--limit',"$Limit") }
    Invoke-Checked $python $a
    # Visual check only: a timestamped directory (reruns never collide) and failures only warn.
    $diag = "$Run/diagnostics_prepare_" + (Get-Date -Format 'yyyyMMdd_HHmmss')
    & $python diagnostics.py --config $Config --manifest "$Run/manifest.json" --output $diag
    if ($LASTEXITCODE -ne 0) { Write-Warning "Contact sheet incomplete (exit $LASTEXITCODE); see $diag/contact_sheet.png.json" }
}
if ($StepList -contains 'extract') {
    & "$PSScriptRoot\extract_dual_gpu.ps1" -EnvDir $EnvDir -Config $Config -Manifest "$Run/manifest.json" -Cache "$Run/features" -GPUs $GPUs
}
if ($StepList -contains 'train') {
    Invoke-Checked $python @('train.py','--config',$Config,'--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--output',"$Run/training")
    Invoke-Checked $python @('export.py','--checkpoint',"$Run/training/best.pth",'--output',"$Run/scc_retrieval_head.pth")
}
if ($StepList -contains 'eval') {
    Invoke-Checked $python @('evaluate.py','--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--checkpoint',"$Run/training/best.pth",'--split','val','--output',"$Run/evaluation_val.json")
}
