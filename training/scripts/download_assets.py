"""Online-only official assets. Runtime never invokes this downloader."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, sha256, write_json

COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
ASSETS = [
    ('weights/dinov2_vitl14_pretrain.pth', 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', None),
    ('installers/Miniconda3-py310_25.1.1-2-Windows-x86_64.exe', 'https://repo.anaconda.com/miniconda/Miniconda3-py310_25.1.1-2-Windows-x86_64.exe', '80ff6cb7cef57be6ff1c7997fad96de3783934e9cea5c46769f840b8bd5b921f'),
    ('installers/vc_redist.x64.exe', 'https://aka.ms/vs/17/release/vc_redist.x64.exe', None),
]

def download(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = Path(str(target)+'.partial')
        for attempt in range(12):
            r = subprocess.run(['curl.exe', '-sS', '--fail', '--location', '--continue-at', '-', '--output', str(tmp), url])
            if r.returncode == 0: tmp.replace(target); break
        if not target.exists(): raise RuntimeError('Download incomplete: '+url)
    actual = sha256(target)
    if expected and actual != expected: raise ValueError('Official SHA256 mismatch: '+str(target))
    return {'url': url, 'filename': target.name, 'sha256': actual, 'official_hash_checked': bool(expected), 'bytes': target.stat().st_size}

def main(bundle):
    records = []
    for rel, url, expected in ASSETS:
        target = ROOT/rel if rel.startswith('weights/') else Path(bundle)/rel
        records.append(download(url, target, expected))
    archive = Path(bundle)/'source_archives'/f'dinov2-{COMMIT}.zip'
    records.append(download(f'https://codeload.github.com/facebookresearch/dinov2/zip/{COMMIT}', archive))
    dest = ROOT/'vendor/dinov2'
    if not dest.exists():
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                parts = Path(entry.filename).parts[1:]
                if not parts: continue
                target = (dest/Path(*parts)).resolve()
                if not target.is_relative_to(dest.resolve()): raise ValueError('Unsafe ZIP')
                if entry.is_dir(): target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(z.read(entry))
    (dest/'SOURCE_COMMIT.txt').write_text(COMMIT)
    write_json(Path(bundle)/'asset_sources.json', {'source_commit': COMMIT, 'assets': records,
               'backbone_status': 'Official alternative. Actual deployed MINIMA shared backbone identity NOT confirmed.'})

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', required=True); a = p.parse_args(); main(a.bundle)
