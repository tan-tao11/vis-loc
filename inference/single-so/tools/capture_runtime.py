#!/usr/bin/env python3
"""Capture the validated ARM64 environment, excluding all task weights and data."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--source-home',default=str(Path.home()));a=ap.parse_args()
SOURCE_HOME=Path(a.source_home)
stage=ROOT/'runtime-stage'
if stage.exists():raise SystemExit('runtime-stage exists; use a new build directory or explicitly remove the previous staging tree')
stage.mkdir()
def linkcopy(src,dst):
    try:os.link(src,dst)
    except OSError:
        shutil.copyfile(src,dst)
        shutil.copymode(src,dst)
    return str(dst)
def env_ignore(folder,names):
    return [n for n in names if n in ('__pycache__','.git') or n.endswith(('.pyc','.a')) or
            (Path(folder)==SOURCE_HOME/'.venvs/uavloc' and n in ('cache','include','bin','share'))]
def project_ignore(folder,names):
    return [n for n in names if n in ('.git','__pycache__','assets','Result','data') or
            Path(n).suffix.lower() in ('.pth','.pt','.ckpt','.onnx','.safetensors','.jpg','.png','.jpeg','.gif','.mp4','.zip','.7z','.npz','.npy','.pyc')]
shutil.copytree(SOURCE_HOME/'.venvs/uavloc',stage/'env',symlinks=True,ignore=env_ignore,copy_function=linkcopy)
for name in ('pyvenv.cfg','.gitignore','CACHEDIR.TAG'):
    (stage/'env'/name).unlink(missing_ok=True)
for name in ('AnyVisLoc-Benchmark','SCC-Loc'):
    shutil.copytree(SOURCE_HOME/name,stage/'projects'/name,symlinks=True,ignore=project_ignore,copy_function=linkcopy)
# All source-home absolute links in the virtual environment must become relative.
for p in (stage/'env').rglob('*'):
    if p.is_symlink():
        target=os.readlink(p)
        if target.startswith(str(SOURCE_HOME/'.venvs/uavloc')+'/'):
            dest=stage/'env'/Path(target).relative_to(SOURCE_HOME/'.venvs/uavloc')
            p.unlink();p.symlink_to(os.path.relpath(dest,p.parent))
        elif target.startswith('/'):
            raise RuntimeError('unexpected absolute link: '+str(p)+' -> '+target)
(stage/'python/bin').mkdir(parents=True)
linkcopy('/usr/bin/python3.10',stage/'python/bin/python3.10')
shutil.copytree('/usr/lib/python3.10',stage/'python/lib/python3.10',symlinks=True,copy_function=linkcopy,
                ignore=lambda d,ns:[n for n in ns if n in ('__pycache__','dist-packages','site-packages','sitecustomize.py') or n.startswith('config-3.10-') or n.endswith('.pyc')])
# A copy, not a hardlink, allows subsequent source edits without mutating snapshots.
shutil.copytree(ROOT/'python',stage/'adapter',ignore=shutil.ignore_patterns('__pycache__'))
subprocess.run([str(SOURCE_HOME/'.venvs/uavloc/bin/python'),'-m','pip','install','--no-index','--no-deps',
                '--target',str(stage/'extra'),*map(str,sorted((ROOT/'wheelhouse').glob('*.whl')))],check=True)
(stage/'native').mkdir()
host_required={}
system_libs={}
skip=re.compile(r'^(?:ld-linux.*|lib(?:c|m|dl|pthread|rt|resolv|util|anl|nss_[^.]+)\.so(?:\..*)?)$')
env=dict(os.environ,LD_LIBRARY_PATH=str(SOURCE_HOME/'.venvs/uavloc/lib/python3.10/site-packages/torch/lib'))
def dependencies(path):
    r=subprocess.run(['ldd',str(path)],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    return path,r.stdout
elfs=[stage/'python/bin/python3.10']
for parent in ('env','python','extra'):
    for p in (stage/parent).rglob('*'):
        if p.is_file() and not p.is_symlink() and '.so' in p.name:
            with p.open('rb') as f:
                if f.read(4)==b'\x7fELF':elfs.append(p)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for p,out in ex.map(dependencies,elfs):
        for line in out.splitlines():
            match=re.search(r'^\s*(\S+) => (/\S+)',line)
            if not match:continue
            soname,location=match.groups();source=Path(location)
            if str(source).startswith((str(stage)+'/',str(SOURCE_HOME/'.venvs/uavloc')+'/')):continue
            if skip.match(soname) or soname.startswith(('libcuda.so','libnvidia-','libnvos','libnvrm','libnvcucompat')) or '/nvidia/' in str(source):
                host_required[soname]=str(source);continue
            if soname in system_libs and Path(system_libs[soname]).resolve()!=source.resolve():
                raise RuntimeError('conflicting native dependency '+soname)
            system_libs[soname]=str(source)
# Runtime-loaded CUDA components may not be exposed by ldd.
cuda=Path('/usr/local/cuda/targets/aarch64-linux/lib')
for pattern in ('libnvrtc.so.12','libnvrtc-builtins.so.*'):
    for p in cuda.glob(pattern):system_libs[p.name]=str(p)
for soname,source in system_libs.items():linkcopy(Path(source).resolve(),stage/'native'/soname)
(stage/'system-dependencies.json').write_text(json.dumps({'bundled':system_libs,'host_required':host_required},indent=2))
verify='''import hashlib,json,os,sys\nfrom pathlib import Path\nr=Path(__file__).resolve().parent\nfor e in json.loads((r/'runtime-manifest.json').read_text()):\n p=r/e['path']\n if e['type']=='link':\n  assert p.is_symlink() and os.readlink(p)==e['target'],str(p)\n else:\n  assert p.is_file() and not p.is_symlink(),str(p)\n  h=hashlib.sha256()\n  with p.open('rb') as f:\n   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)\n  assert h.hexdigest()==e['sha256'],str(p)\nprint('RUNTIME_MANIFEST_OK',flush=True)\n'''
(stage/'verify_runtime.py').write_text(verify)
entries=[]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
for p in sorted(stage.rglob('*')):
    rel=p.relative_to(stage).as_posix()
    if p.is_symlink():
        assert not os.readlink(p).startswith('/'),rel
        assert p.exists(),rel
        entries.append(dict(path=rel,type='link',target=os.readlink(p)))
    elif p.is_file():entries.append(dict(path=rel,type='file',size=p.stat().st_size,sha256=digest(p)))
(stage/'runtime-manifest.json').write_text(json.dumps(entries,indent=2))
print('Runtime files:',len(entries),'bytes:',sum(e.get('size',0) for e in entries),flush=True)
with gzip.open(ROOT/'runtime.tar.gz','wb',compresslevel=1) as f:
    with tarfile.open(fileobj=f,mode='w',dereference=False) as tar:
        tar.add(stage,arcname='.',recursive=True)
print('Created',ROOT/'runtime.tar.gz',flush=True)
