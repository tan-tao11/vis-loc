# One-shot MLP pipeline: prepare -> dual-GPU feature extraction -> train/export -> val evaluation. Output dir = config "output".
# Use -Limit 12 for a small check first; -Steps selects stages, e.g. -Steps prepare  or  -Steps extract,train,eval
param([Parameter(Mandatory=$true)][string]$EnvDir, [string]$Config='configs/mlp.yaml', [int]$GPUs=2,
      [int]$Limit=0, [string[]]$Steps=@('prepare','extract','train','eval'))
. "$PSScriptRoot\common.ps1"
Set-EnvironmentPath $EnvDir
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
$python = Join-Path $EnvDir 'python.exe'
$Run = (& $python -c "import sys; from common import config; print(config(sys.argv[1])['output'])" $Config).Trim()
if ($LASTEXITCODE -ne 0 -or -not $Run) { throw "Cannot read output from $Config" }
if ($Steps -contains 'prepare') {
    $a = @('prepare_dataset.py','--config',$Config); if ($Limit -gt 0) { $a += @('--limit',"$Limit") }
    Invoke-Checked $python $a
    Invoke-Checked $python @('diagnostics.py','--config',$Config,'--manifest',"$Run/manifest.json",'--output',"$Run/diagnostics_prepare")
}
if ($Steps -contains 'extract') {
    & "$PSScriptRoot\extract_dual_gpu.ps1" -EnvDir $EnvDir -Config $Config -Manifest "$Run/manifest.json" -Cache "$Run/features" -GPUs $GPUs
}
if ($Steps -contains 'train') {
    Invoke-Checked $python @('train.py','--config',$Config,'--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--output',"$Run/training")
    Invoke-Checked $python @('export.py','--checkpoint',"$Run/training/best.pth",'--output',"$Run/scc_retrieval_head.pth")
}
if ($Steps -contains 'eval') {
    Invoke-Checked $python @('evaluate.py','--manifest',"$Run/manifest.json",'--cache',"$Run/features",'--checkpoint',"$Run/training/best.pth",'--split','val','--output',"$Run/evaluation_val.json")
}
