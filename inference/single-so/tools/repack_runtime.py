#!/usr/bin/env python3
"""Refresh adapter code in an extracted runtime and regenerate integrity records."""
import gzip,hashlib,json,os,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
stage=ROOT/'runtime-stage'
if not stage.is_dir():
    stage.mkdir()
    with tarfile.open(ROOT/'runtime.tar.gz','r:gz') as t:t.extractall(stage)
for p in (ROOT/'python').glob('*.py'):shutil.copyfile(p,stage/'adapter'/p.name)
# Keep both algorithm trees synchronized with this repository snapshot.
projects = ROOT.parent / 'projects'
for project in ('SCC-Loc', 'AnyVisLoc-Benchmark'):
    source = projects / project
    if not source.is_dir():
        raise SystemExit('missing inference project source: ' + str(source))
    shutil.copytree(source, stage/'projects'/project, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git',
                                                 'weights', 'data', 'datasets', 'outputs', 'logs'))
(stage/'python/bin/python3.10').chmod(0o755)
entries=[]
for p in sorted(stage.rglob('*')):
    rel=p.relative_to(stage).as_posix()
    if rel=='runtime-manifest.json' or '__pycache__' in p.parts or 'cache' in p.relative_to(stage).parts:continue
    if p.is_symlink():entries.append(dict(path=rel,type='link',target=os.readlink(p)))
    elif p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        entries.append(dict(path=rel,type='file',size=p.stat().st_size,sha256=h.hexdigest()))
(stage/'runtime-manifest.json').write_text(json.dumps(entries,indent=2))
def filt(t):
    parts=Path(t.name).parts
    return None if '__pycache__' in parts or 'cache' in parts else t
tmp=ROOT/'runtime.tar.gz.partial'
with gzip.open(tmp,'wb',compresslevel=1) as f:
    with tarfile.open(fileobj=f,mode='w',dereference=False) as t:t.add(stage,arcname='.',filter=filt)
tmp.replace(ROOT/'runtime.tar.gz')
print('Repacked',len(entries),'entries',sum(e.get('size',0) for e in entries),'bytes',flush=True)
