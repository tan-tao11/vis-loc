"""Standalone additive MLP DDP addon; contains no backbone weights or user configs."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,sha256

ADDITIONS=['train_mlp_ddp.py','scripts/train_mlp_ddp.ps1','tests/test_mlp_ddp.py','docs/MLP双卡备选方案.md',
    'scripts/install_mlp_ddp_addon.ps1','scripts/package_mlp_ddp_update.py','ddp_runtime.py','tests/__init__.py']
REQUIRED=['common.py','models.py','cache.py','extract_features.py','train.py','train_lora.py','lora.py','lora_data.py',
    'evaluate.py','evaluate_lora.py','export.py','losses.py','outputs.py','diagnostics.py','geometry.py','records.py',
    'preprocess.py','prepare_dataset.py','synthetic.py','integration/retrieval_adapter.py','scripts/common.ps1']


def package(output):
    entries={'payload/'+name:(ROOT/name).read_bytes() for name in ADDITIONS}
    entries['install_mlp_ddp_addon.ps1']=(ROOT/'scripts/install_mlp_ddp_addon.ps1').read_bytes()
    entries['MLP双卡备选方案.md']=(ROOT/'docs/MLP双卡备选方案.md').read_bytes()
    entries['MLP_DDP_BASE_SHA256SUMS.txt']=('\n'.join(f'{sha256(ROOT/f)}  {f}' for f in REQUIRED)+'\n').encode('utf-8')
    validation=ROOT/'artifacts/mlp_ddp_delivery_validation'
    if validation.is_dir():
        entries.update({'validation/'+p.relative_to(validation).as_posix():p.read_bytes() for p in validation.rglob('*') if p.is_file()})
    entries['MLP_DDP_ADDON_SHA256SUMS.txt']=('\n'.join(f'{hashlib.sha256(data).hexdigest()}  {name}' for name,data in sorted(entries.items()))+'\n').encode('utf-8')
    dest=Path(output); dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in entries.items(): z.writestr(name,data)
    print(f'{dest}: {dest.stat().st_size} bytes; SHA256 {sha256(dest)}')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); package(p.parse_args().output)
