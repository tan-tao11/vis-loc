"""Code-only update for existing deployments; preserves all local configurations."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,sha256

def package(output):
    additions=['outputs.py','diagnostics.py','validate_geometry.py','evaluate.py','evaluate_lora.py',
        'train.py','train_lora.py','export.py','export_lora.py','scripts/evaluate.ps1',
        'scripts/package_debug_update.py','scripts/install_debug_update.ps1','scripts/package_delivery.py',
        'scripts/package_lora_update.py','scripts/package_geometry_update.py',
        'tests/test_outputs.py','README_中文.md','docs/几何验证.md','docs/验证状态.md']
    required=['common.py','geometry.py','records.py','preprocess.py','prepare_dataset.py','cache.py',
        'models.py','lora.py','lora_data.py','losses.py','scripts/common.ps1']
    entries={'payload/'+f:(ROOT/f).read_bytes() for f in additions}
    entries['install_debug_update.ps1']=(ROOT/'scripts/install_debug_update.ps1').read_bytes()
    entries['DEBUG_BASE_SHA256SUMS.txt']=('\n'.join(f'{sha256(ROOT/f)}  {f}' for f in required)+'\n').encode('utf-8')
    entries['DEBUG_UPDATE_SHA256SUMS.txt']=('\n'.join(f'{hashlib.sha256(data).hexdigest()}  {name}' for name,data in sorted(entries.items()))+'\n').encode('utf-8')
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in entries.items(): z.writestr(name,data)
    print(f'{out}: {out.stat().st_size} bytes; SHA256 {sha256(out)}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);package(p.parse_args().output)
