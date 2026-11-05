from pathlib import Path
import numpy as np
from common import read_json, digest, verify_manifest

def group_cache_name(identity): return digest(identity)+'.npz'

def load_cache(manifest, cache_dir, verify=True):
    m = read_json(manifest) if not isinstance(manifest, dict) else manifest
    if m['leakage']: raise ValueError('Manifest contains split leakage')
    if verify: verify_manifest(m)
    cache_dir = Path(cache_dir); index = read_json(cache_dir/'index.json')
    if index['manifest_id'] != m['manifest_id']: raise ValueError('Cache/manifest mismatch')
    if verify:
        from models import backbone_identity
        from extract_features import extraction_identity
        current = extraction_identity(m, m['config'], backbone_identity(m['config']))
        if current != index['fingerprint']: raise ValueError('Model/source/preprocessing changed; rebuild feature cache')
    if index['group_ids'] != [g['id'] for g in m['groups']]: raise ValueError('Cache group ordering/integrity mismatch')
    arrays = {}
    for g in m['groups']:
        with np.load(cache_dir/group_cache_name(g['id']), allow_pickle=False) as f:
            if str(f['fingerprint']) != index['fingerprint']: raise ValueError('Stale cache')
            if str(f['id']) != g['id']: raise ValueError('Wrong group cache identity')
            q, s = f['query'].copy(), f['satellite'].copy()
            if q.shape != (1024,) or s.shape != (len(g['candidates']), 1024): raise ValueError('Feature shape mismatch')
            if not np.isfinite(q).all() or not np.isfinite(s).all(): raise ValueError('Nonfinite cached feature')
            arrays[g['id']] = (q, s)
    return m, arrays, index
