"""Additive geometry diagnostic update; no model weights or data config."""
import argparse
import hashlib
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, sha256


def package(output):
    additions = ['validate_geometry.py', 'outputs.py', 'scripts/validate_geometry.ps1', 'docs/几何验证.md', 'tests/test_geometry_validation.py']
    required = ['common.py', 'records.py', 'geometry.py', 'preprocess.py', 'scripts/common.ps1']
    entries = {name: (ROOT/name).read_bytes() for name in additions}
    entries['GEOMETRY_BASE_SHA256SUMS.txt'] = ('\n'.join(f'{sha256(ROOT/name)}  {name}' for name in required)+'\n').encode('utf-8')
    entries['GEOMETRY_SHA256SUMS.txt'] = ('\n'.join(
        f'{hashlib.sha256(data).hexdigest()}  {name}' for name, data in sorted(entries.items()))+'\n').encode('utf-8')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items(): archive.writestr(name, data)
    print(f'{output}: {output.stat().st_size} bytes; SHA256 {sha256(output)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    package(p.parse_args().output)
