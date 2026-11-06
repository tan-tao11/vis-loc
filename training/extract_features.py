import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from common import config, path, read_json, write_json, digest, verify_manifest, sha256, ROOT, require_usable_device
from cache import group_cache_name
from preprocess import query_image, tensor_image
from scc_inference import TileSource, tile_tensor
from models import load_backbone, forward_frozen, backbone_identity

def extraction_identity(m, c, identity):
    return digest({'manifest_id': m['manifest_id'], 'config': c, 'backbone': identity,
                   'pipeline_code': {name: sha256(ROOT/name) for name in ('geometry.py', 'preprocess.py', 'scc_inference.py', 'scc_geodata.py', 'scc_oblique.py', 'models.py', 'extract_features.py')},
                   'pipeline_version': 1, 'torch': str(torch.__version__), 'amp': c['extract']['amp']})

def extract(c, manifest, cache_dir, device='cuda:0', shard_index=0, num_shards=1):
    require_usable_device(device)
    m = read_json(manifest); verify_manifest(m)
    if digest(c) != m['fingerprint']['config']: raise ValueError('Configuration changed after preparation')
    if not 0 <= shard_index < num_shards: raise ValueError('Invalid shard partition')
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    backbone, identity = load_backbone(c, device)
    fp = extraction_identity(m, c, identity)
    tiles = TileSource(c); pp = c['preprocess']
    failures = []; chosen = [g for i, g in enumerate(m['groups']) if i % num_shards == shard_index]
    batch_size = c['extract']['batch_size']; start = time.perf_counter()
    # Real input memory probe; halve on CUDA OOM, then keep measured safe batch.
    probe = torch.zeros(1, 3, c['preprocess']['size'], c['preprocess']['size'], device=device)
    while True:
        try:
            forward_frozen(backbone, probe.expand(batch_size, -1, -1, -1), c['extract']['amp'])
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1: raise
            batch_size //= 2; torch.cuda.empty_cache()
    del probe
    try:
        for g in chosen:
            dest = cache_dir/group_cache_name(g['id'])
            if dest.exists():
                try:
                    with np.load(dest, allow_pickle=False) as f:
                        if str(f['fingerprint']) == fp and str(f['id']) == g['id'] and f['satellite'].shape == (len(g['candidates']), 1024) and np.isfinite(f['satellite']).all() and np.isfinite(f['query']).all(): continue
                except (ValueError, OSError): pass
            for attempt in range(c['extract']['retries']+1):
                try:
                    t = time.perf_counter()
                    u = tiles.query(g, pp['size'])
                    q = forward_frozen(backbone, tensor_image(u, c['preprocess']).unsqueeze(0).to(device), c['extract']['amp'])[0].cpu().numpy()
                    feats = []
                    for j in range(0, len(g['candidates']), batch_size):
                        images = [tile_tensor(tiles.tile(g, k, pp['size']), pp['mean'], pp['std']) for k in range(j, min(j+batch_size, len(g['candidates'])))]
                        feats.append(forward_frozen(backbone, torch.stack(images).to(device), c['extract']['amp']).cpu().numpy())
                    sat = np.concatenate(feats) if feats else np.empty((0, 1024), np.float32)
                    with open(str(dest)+'.partial', 'wb') as f:
                        np.savez(f, id=g['id'], fingerprint=fp, query=q, satellite=sat, seconds=time.perf_counter()-t)
                    os.replace(str(dest)+'.partial', dest)
                    print(f'{g["id"]}: {len(sat)} candidates, batch={batch_size}', flush=True)
                    break
                except torch.cuda.OutOfMemoryError:
                    batch_size = max(1, batch_size//2); torch.cuda.empty_cache()
                    if attempt == c['extract']['retries']: failures.append({'id': g['id'], 'error': 'CUDA OOM'})
                except Exception as e:
                    if attempt == c['extract']['retries']: failures.append({'id': g['id'], 'error': repr(e)})
    finally:
        tiles.close()
    write_json(cache_dir/f'shard_{shard_index}.json', {'shard_index': shard_index, 'num_shards': num_shards,
        'assigned_ids': [g['id'] for g in chosen], 'fingerprint': fp, 'backbone': identity,
        'batch_size': batch_size, 'device': device, 'failures': failures, 'seconds': time.perf_counter()-start})
    if failures: raise RuntimeError('Feature extraction failures; see shard report')

def merge(c, manifest, cache_dir, num_shards):
    m = read_json(manifest); verify_manifest(m); directory = Path(cache_dir)
    identity = backbone_identity(c); fp = extraction_identity(m, c, identity)
    assigned = []; elapsed = 0.
    for i in range(num_shards):
        r = read_json(directory/f'shard_{i}.json')
        if r['fingerprint'] != fp or r['num_shards'] != num_shards or r['failures']: raise ValueError('Stale/incomplete shard')
        expected = [g['id'] for j, g in enumerate(m['groups']) if j % num_shards == i]
        if r['assigned_ids'] != expected: raise ValueError('Shard partition mismatch')
        assigned += r['assigned_ids']
    ids = [g['id'] for g in m['groups']]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(ids): raise ValueError('Duplicate/missing shard work')
    for g in m['groups']:
        with np.load(directory/group_cache_name(g['id']), allow_pickle=False) as f:
            if str(f['fingerprint']) != fp or str(f['id']) != g['id']: raise ValueError('Cache fingerprint mismatch')
            if f['query'].shape != (1024,) or f['satellite'].shape != (len(g['candidates']), 1024): raise ValueError('Cache shape mismatch')
            if not np.isfinite(f['query']).all() or not np.isfinite(f['satellite']).all(): raise ValueError('Nonfinite cache')
            elapsed += float(f['seconds'])
    write_json(directory/'index.json', {'manifest_id': m['manifest_id'], 'fingerprint': fp, 'backbone': identity,
        'group_ids': ids, 'extract_seconds_sum': elapsed, 'num_shards': num_shards})

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--manifest', required=True)
    p.add_argument('--cache', required=True); p.add_argument('--device', default='cuda:0'); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1); p.add_argument('--merge', action='store_true'); a = p.parse_args(); c = config(a.config)
    if a.merge: merge(c, a.manifest, a.cache, a.num_shards)
    else:
        extract(c, a.manifest, a.cache, a.device, a.shard_index, a.num_shards)
        if a.num_shards == 1: merge(c, a.manifest, a.cache, 1)
