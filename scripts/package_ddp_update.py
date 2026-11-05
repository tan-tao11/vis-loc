"""Additive-only optional DDP package; never rewrites the single-GPU deployment."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,sha256

ADDITIONS=['ddp_runtime.py','train_lora_ddp.py','ddp_check.py','ddp_smoke.py','configs/lora_ddp.yaml',
    'scripts/train_lora_ddp.ps1','scripts/check_ddp.ps1','scripts/install_ddp_addon.ps1','scripts/package_ddp_update.py',
    'tests/__init__.py','tests/test_ddp.py','docs/双卡备选方案.md']
REQUIRED=['common.py','models.py','lora.py','lora_data.py','train_lora.py','evaluate.py','evaluate_lora.py','export_lora.py',
    'outputs.py','diagnostics.py','geometry.py','records.py','preprocess.py','prepare_dataset.py','synthetic.py','losses.py',
    'integration/lora_adapter.py','scripts/common.ps1','tests/test_lora.py']

def package(output):
    entries={'payload/'+name:(ROOT/name).read_bytes() for name in ADDITIONS}
    entries['install_ddp_addon.ps1']=(ROOT/'scripts/install_ddp_addon.ps1').read_bytes()
    entries['双卡备选方案.md']=(ROOT/'docs/双卡备选方案.md').read_bytes()
    entries['DDP_BASE_SHA256SUMS.txt']=('\n'.join(f'{sha256(ROOT/f)}  {f}' for f in REQUIRED)+'\n').encode('utf-8')
    validation=ROOT/'artifacts/ddp_delivery_validation'
    if validation.is_dir():
        entries.update({'validation/'+p.relative_to(validation).as_posix():p.read_bytes() for p in validation.rglob('*') if p.is_file()})
    entries['DDP_ADDON_SHA256SUMS.txt']=('\n'.join(f'{hashlib.sha256(data).hexdigest()}  {name}' for name,data in sorted(entries.items()))+'\n').encode('utf-8')
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in entries.items(): z.writestr(name,data)
    print(f'{dest}: {dest.stat().st_size} bytes; SHA256 {sha256(dest)}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);package(p.parse_args().output)
