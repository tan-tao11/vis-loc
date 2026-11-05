"""Shared configuration, strict hashes and atomic portable artifacts."""
from pathlib import Path
import hashlib
import json
import os
import random
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent

def path(value):
    if value is None:
        raise ValueError('Required path is not configured')
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

# Mappings replaced wholesale when overridden (their keys are data names, not settings).
REPLACE_KEYS = ('sequence_splits', 'sequence_by_file', 'groups')

def merge_config(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = merge_config(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) and k not in REPLACE_KEYS else v
    return out

def config(filename):
    """Load YAML; `base: other.yaml` (relative to this file) supplies every omitted setting."""
    with open(filename, encoding='utf-8-sig') as f:
        c = yaml.safe_load(f)
    base = c.pop('base', None)
    if base:
        c = merge_config(config(Path(filename).parent/base), c)
    if c['schema_version'] != 1:
        raise ValueError('Unsupported config schema')
    return c

def sha256(filename):
    h = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()

def write_json(filename, obj, indent=2):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.partial')
    t.write_text(json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(t, p)

def read_json(filename):
    return json.loads(Path(filename).read_text(encoding='utf-8'))

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def require_real_ready(c):
    if c.get('synthetic'):
        return
    checks = [(c['data']['mapping_confirmed'], 'TXT/image mapping'),
              (c['pose']['calibration_confirmed'], 'camera installation calibration'),
              (c['geo']['vertical_confirmed'], 'vertical datum'),
              (c['preprocess']['deployment_confirmed'], 'deployment preprocessing')]
    missing = [name for ok, name in checks if not ok]
    if missing:
        raise ValueError('Formal labels blocked; confirmation missing: ' + ', '.join(missing))

def raster_files(filename):
    p = path(filename)
    return [p] + [p.with_suffix(s) for s in ('.tfw', '.prj', '.aux.xml') if p.with_suffix(s).exists()]

def input_fingerprint(c, files):
    return {'config': digest(c), 'files': {str(Path(f).resolve()): sha256(f) for f in sorted(set(files), key=str)}}

def verify_inputs(fp):
    for p, h in fp['files'].items():
        if not Path(p).is_file() or sha256(p) != h:
            raise ValueError('Input changed or missing; rebuild manifest/cache: ' + p)

GENERATOR_FILES = ('records.py', 'geometry.py', 'preprocess.py', 'prepare_dataset.py', 'scc_inference.py', 'scc_geodata.py')

def generator_identity():
    return {name: sha256(ROOT/name) for name in GENERATOR_FILES}

def verify_manifest(m):
    if digest({k: v for k, v in m.items() if k != 'manifest_id'}) != m['manifest_id']:
        raise ValueError('Manifest content integrity mismatch')
    if m.get('generator_code') != generator_identity():
        raise ValueError('Dataset preparation code changed; regenerate manifest')
    if digest(m['config']) != m['fingerprint']['config']:
        raise ValueError('Manifest configuration mismatch')
    verify_inputs(m['fingerprint'])
