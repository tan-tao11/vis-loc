"""Build offline code/environment assets; DINO weights are supplied by the server."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, sha256, write_json


@contextmanager
def atomic_zip(filename, compression):
    """Keep an existing deliverable intact until its replacement closes successfully."""
    filename=Path(filename); temporary=filename.with_name(filename.name+'.partial')
    try:
        with zipfile.ZipFile(temporary,'w',compression) as archive: yield archive
        os.replace(temporary,filename)
    finally: temporary.unlink(missing_ok=True)

def hashes(root, output):
    root = Path(root).resolve(); output = Path(output).resolve()
    entries = [p for p in root.rglob('*') if p.is_file() and p != output and '.git' not in p.parts and '__pycache__' not in p.parts and not p.name.endswith('.partial')]
    output.write_text('\n'.join(f'{sha256(p)}  {p.relative_to(root).as_posix()}' for p in sorted(entries))+'\n', encoding='utf-8')
    return len(entries)

def package(env, bundle, skip_env=False):
    env = Path(env).resolve(); bundle = Path(bundle).resolve(); bundle.mkdir(parents=True, exist_ok=True)
    if Path(sys.prefix).resolve() != env: raise ValueError('Run packaging using target environment python')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    wheels = bundle/'wheelhouse'
    from packaging.utils import parse_wheel_filename, canonicalize_name
    from packaging.tags import sys_tags
    valid_tags = set(sys_tags()); available = {}
    for wh in wheels.glob('*.whl'):
        name, version, build, tags = parse_wheel_filename(wh.name)
        if not tags & valid_tags: raise ValueError('Wrong OS/ABI wheel: '+wh.name)
        available[(canonicalize_name(name), str(version))] = wh
    lock = []; wheel_sources = []
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name, version = dist.metadata['Name'], dist.version
        wh = available.get((canonicalize_name(name), version))
        if wh is None: raise ValueError(f'Missing exact wheel for installed dependency: {name}=={version}')
        lock.append(f'{name}=={version} --hash=sha256:{sha256(wh)}')
        wheel_sources.append({'package': name, 'version': version, 'wheel': wh.name, 'sha256': sha256(wh),
                              'index': 'https://download.pytorch.org/whl/cu118' if name in ('torch', 'torchvision') else 'https://pypi.org/simple'})
    (bundle/'requirements.lock.txt').write_text('\n'.join(lock)+'\n', encoding='utf-8')
    conda_dir = bundle/'conda_packages'; conda_dir.mkdir(exist_ok=True); packages = []
    for file in sorted((env/'conda-meta').glob('*.json')):
        meta = json.loads(file.read_text()); archive = Path(meta['link']['source']).parent/meta['fn']
        if not archive.is_file(): raise FileNotFoundError('Conda archive missing: '+str(archive))
        target = conda_dir/archive.name
        shutil.copy2(archive, target)
        if meta.get('sha256') and sha256(target) != meta['sha256']: raise ValueError('Conda archive hash mismatch')
        packages.append({'filename': target.name, 'url': meta['url'], 'sha256': sha256(target), 'version': meta['version'], 'build': meta['build']})
    write_json(bundle/'conda_explicit.json', packages); write_json(bundle/'wheel_sources.json', wheel_sources)
    provenance_file = bundle/'asset_sources.json'
    provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
    for asset in provenance['assets']:
        if asset['filename'] == 'dinov2_vitl14_pretrain.pth':
            asset['included_in_deployment'] = False
            asset['availability'] = 'Server-provided; reference hash records preparation-machine validation only'
    provenance['backbone_status'] = 'DINO checkpoint excluded from deployment. Configure the existing server checkpoint; actual SCC backbone identity remains unverified.'
    provenance['model_payload'] = {'archive': 'model_and_source.zip', 'contains_dino_weights': False,
        'contents': 'Pinned DINOv2 source and licenses only; historical archive filename retained',
        'server_configuration': ['configs/train.yaml: model.checkpoint', 'configs/train.yaml: model.checkpoint_sha256']}
    write_json(provenance_file, provenance)
    # The full installer restores this integrated source directly; no patch chain.
    allowed_dirs = ['configs', 'integration', 'scripts', 'tests', 'docs', 'reference']
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.md', '.in', '.ini', '.txt') and p.name != 'SHA256SUMS.txt']
    files += [p for d in allowed_dirs for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    required=['train.py','train_mlp_ddp.py','train_lora.py','train_lora_ddp.py','train_lora_aug.py',
        'train_lora_aug_ddp.py','appearance_augmentation.py','ddp_runtime.py','validate_geometry.py','outputs.py',
        'scripts/train_mlp_ddp.ps1','scripts/train_lora_ddp.ps1','scripts/train_lora_aug.ps1','scripts/train_lora_aug_ddp.ps1',
        'configs/lora_aug.yaml','configs/lora_aug_ddp.yaml']
    source_files={p.relative_to(ROOT).as_posix():p for p in files}
    if not set(required)<=set(source_files): raise ValueError('Missing integrated feature: '+str(set(required)-set(source_files)))
    if any(p.suffix.lower() in ('.pth','.pt','.ckpt','.safetensors') for p in files): raise ValueError('Weight file in source payload')
    release={'format_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'integrated_install':True,
        'requires_addon_installation':False,'dino_weights_included':False,
        'features':['mlp_single','mlp_ddp','lora_single','lora_ddp','lora_augmentation_single','lora_augmentation_ddp',
                    'geometry_validation','evaluation_html_csv_images','training_and_augmentation_previews'],
        'python':sys.version.split()[0],
        'torch':importlib.metadata.version('torch'),
        'source_sha256':{name:sha256(p) for name,p in sorted(source_files.items())}}
    write_json(bundle/'release_manifest.json',release)
    with atomic_zip(bundle/'source.zip', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files): z.write(f, f.relative_to(ROOT).as_posix())
        z.write(bundle/'release_manifest.json','release_manifest.json')
    with atomic_zip(bundle/'model_and_source.zip', zipfile.ZIP_STORED) as z:
        # Keep this filename for installer compatibility, but never ship the local
        # weights/ directory or a cached checkpoint from the vendored source tree.
        for f in (ROOT/'vendor/dinov2').rglob('*'):
            if (f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts
                    and not f.name.endswith('.partial') and f.suffix.lower() not in ('.pth', '.pt', '.ckpt', '.safetensors')):
                z.write(f, f.relative_to(ROOT).as_posix())
    if not skip_env:
        import conda_pack
        conda_pack.pack(prefix=str(env), output=str(bundle/'environment_win64_py310_cu118.zip'), format='zip', force=True)
    shutil.copy2(ROOT/'scripts/offline_install.ps1', bundle/'offline_install.ps1')
    shutil.copy2(ROOT/'scripts/common.ps1', bundle/'common.ps1')
    for name in ('README_中文.md', 'docs/验证状态.md'):
        if (ROOT/name).is_file(): shutil.copy2(ROOT/name, bundle/Path(name).name)
    # Standalone backup ZIP has its own hash manifest; it does not require archive A.
    backup_files = [bundle/name for name in ('source.zip', 'model_and_source.zip', 'release_manifest.json', 'offline_install.ps1', 'common.ps1',
        'requirements.lock.txt', 'conda_explicit.json', 'asset_sources.json', 'wheel_sources.json', 'README_中文.md', '验证状态.md')]
    backup_files += [f for directory in ('wheelhouse', 'conda_packages', 'installers', 'validation') for f in (bundle/directory).rglob('*')
        if f.is_file() and f.suffix.lower() not in ('.pth','.pt','.ckpt','.safetensors')]
    backup_hashes = '\n'.join(f'{sha256(f)}  {f.relative_to(bundle).as_posix()}' for f in sorted(backup_files))+'\n'
    with atomic_zip(bundle/'offline_rebuild_B.zip', zipfile.ZIP_STORED) as z:
        for f in backup_files: z.write(f, f.relative_to(bundle).as_posix())
        z.writestr('SHA256SUMS.txt', backup_hashes)
    hashes(bundle, bundle/'SHA256SUMS.txt')
    print('Bundle written:', bundle)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--env', required=True); p.add_argument('--bundle', required=True); p.add_argument('--skip-env', action='store_true')
    a = p.parse_args(); package(a.env, a.bundle, a.skip_env)
