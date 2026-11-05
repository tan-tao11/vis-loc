"""Small additive update, with required base-code hashes; data config is excluded."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, sha256


def package(output):
    additions = [
        'lora.py', 'lora_data.py', 'train_lora.py', 'evaluate_lora.py', 'export_lora.py', 'lora_smoke.py', 'outputs.py', 'diagnostics.py',
        'configs/lora.yaml', 'integration/lora_adapter.py', 'integration/feature_extract.lora.snapshot.patch',
        'scripts/train_lora.ps1', 'scripts/evaluate_lora.ps1', 'scripts/lora_smoke.ps1',
        'scripts/make_lora_integration_patch.py', 'tests/test_lora.py', 'docs/LoRA备选方案.md']
    required = ['common.py', 'records.py', 'geometry.py', 'preprocess.py', 'prepare_dataset.py',
        'models.py', 'losses.py', 'evaluate.py', 'cache.py', 'extract_features.py', 'synthetic.py', 'offline_check.py', 'scripts/common.ps1']
    entries = {'payload/'+name: (ROOT/name).read_bytes() for name in additions}
    entries['install_lora_update.ps1'] = (ROOT/'scripts/install_lora_update.ps1').read_bytes()
    entries['LORA_BASE_SHA256SUMS.txt'] = ('\n'.join(f'{sha256(ROOT/name)}  {name}' for name in required)+'\n').encode('utf-8')
    entries['LORA_UPDATE_SHA256SUMS.txt'] = ('\n'.join(
        f'{hashlib.sha256(data).hexdigest()}  {name}' for name, data in sorted(entries.items()))+'\n').encode('utf-8')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items(): archive.writestr(name, data)
    print(f'{output}: {output.stat().st_size} bytes; SHA256 {sha256(output)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    package(p.parse_args().output)
