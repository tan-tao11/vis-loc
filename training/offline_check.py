"""Offline diagnostics with Python socket egress blocked, explicit GPU assertions."""
import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import traceback
from common import config, sha256, write_json, ROOT

@contextlib.contextmanager
def deny_network():
    original_connect = socket.socket.connect; original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection
    def blocked(*args, **kwargs): raise RuntimeError('Network access forbidden during offline validation')
    socket.socket.connect = blocked; socket.socket.connect_ex = blocked; socket.create_connection = blocked
    try: yield
    finally:
        socket.socket.connect = original_connect; socket.socket.connect_ex = original_connect_ex; socket.create_connection = original_create

def check_hashes(root, checksum_file):
    root = Path(root).resolve(); checked = 0
    for line in Path(checksum_file).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        expected, name = line.split(None, 1); p = (root/name).resolve()
        if not p.is_relative_to(root): raise ValueError('Unsafe checksum path')
        if sha256(p) != expected.lower(): raise ValueError('SHA256 mismatch: '+name)
        checked += 1
    if not checked: raise ValueError('Empty checksum manifest')
    return checked

def run(args):
    report = {'python': sys.version, 'platform': platform.platform(), 'checks': {}, 'errors': [],
              'network_scope': 'Python socket connect/create_connection blocked; no OS-wide firewall changes'}
    def check(name, fn):
        try:
            report['checks'][name] = {'passed': True, 'result': fn()}; print('PASS', name, flush=True)
        except Exception as e:
            report['checks'][name] = {'passed': False, 'error': repr(e)}; report['errors'].append(name)
            print('FAIL', name, repr(e), flush=True)
    if args.hashes:
        check('artifact_hashes', lambda: check_hashes(args.bundle_root, args.hashes))
    else:
        report['checks']['artifact_hashes'] = {'passed': False, 'error': 'No hashes supplied'}
        report['errors'].append('artifact_hashes')
    def imports():
        return {m: getattr(importlib.import_module(m), '__version__', 'loaded') for m in
                ['torch', 'torchvision', 'numpy', 'cv2', 'PIL', 'rasterio', 'pyproj', 'shapely', 'yaml', 'tqdm']}
    check('dependencies_and_DLLs', imports)
    if args.expected_gpus:
        check('nvidia_driver_inventory', lambda: subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip())
    import torch
    def gpus():
        if not torch.cuda.is_available() and args.expected_gpus: raise RuntimeError('CUDA unavailable')
        n = torch.cuda.device_count()
        if n < args.expected_gpus: raise RuntimeError(f'Expected {args.expected_gpus} GPUs, found {n}')
        result = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            if args.require_a4000 and ('A4000' not in props.name or props.total_memory < 15*1024**3): raise RuntimeError('Not a 16GB RTX A4000')
            x = torch.randn(256, 256, device=f'cuda:{i}', requires_grad=True); (x@x.T).square().mean().backward()
            if not torch.isfinite(x.grad).all(): raise RuntimeError('GPU compute nonfinite')
            torch.cuda.synchronize(i)
            result.append({'index': i, 'name': props.name, 'bytes': props.total_memory})
        return result
    check('independent_GPU_compute', gpus)
    def pipeline():
        from synthetic import make_fixture
        from prepare_dataset import prepare
        from extract_features import extract, merge
        from train import train
        from evaluate import evaluate_arrays, load_heads_checkpoint
        from cache import load_cache
        from export import export
        from diagnostics import bice_examples, contact_sheet
        from integration.retrieval_adapter import RetrievalAdapter
        root = Path(args.work).resolve(); cf = make_fixture(root); c = config(cf)
        # Local source/weights from requested config; synthetic data is generated in this new path.
        base = config(args.config); c['model'] = base['model']; c['train']['epochs'] = 3; c['train']['patience'] = 3
        import yaml
        cf.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding='utf-8')
        manifest = prepare(c); mf = root/'run/manifest.json'; cache = root/'features'
        extract(c, mf, cache, args.device); merge(c, mf, cache, 1)
        best = train(c, mf, cache, root/'training', args.device)
        heads, ck = load_heads_checkpoint(best, args.device); m, arrays, index = load_cache(mf, cache)
        validation = evaluate_arrays(m, arrays, heads, 'val', args.device); write_json(root/'validation.json', validation)
        test = evaluate_arrays(m, arrays, heads, 'test', args.device); write_json(root/'test_final.json', test)
        outfile = root/'synthetic_ONLY_scc_retrieval_head.pth'; meta = export(best, outfile)
        adapter = RetrievalAdapter(outfile, True, args.device, meta['backbone']['checkpoint_sha256'], meta['preprocess_pose_fingerprint'], allow_synthetic=True)
        q = torch.randn(3, 1024, device=args.device)
        for domain in ('ir', 'satellite'):
            if not torch.equal(heads(q, domain), adapter.apply(q, domain)): raise ValueError('Export reload numerical mismatch')
        bice_examples(c, root/'bice_synthetic_plane.json'); contact_sheet(mf, root/'contact_sheet.png', root/'validation.json')
        return {'queries': len(manifest['groups']), 'candidates': sum(len(g['candidates']) for g in manifest['groups']),
                'backbone': index['backbone'], 'output': str(root), 'synthetic_only': True}
    with deny_network():
        check('network_guard', lambda: network_guard_test())
        check('local_DINO_prepare_extract_train_evaluate_export_reload', pipeline)
    report['passed'] = not report['errors']
    write_json(args.output, report)
    return 0 if report['passed'] else 1

def network_guard_test():
    try: socket.create_connection(('example.com', 443))
    except RuntimeError: return 'Outbound connect blocked as expected'
    raise RuntimeError('Network guard failed')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--expected-gpus', type=int, default=2); p.add_argument('--require-a4000', action='store_true')
    p.add_argument('--hashes'); p.add_argument('--bundle-root', default='.'); p.add_argument('--work', default='artifacts/offline_smoke')
    p.add_argument('--output', default='artifacts/offline_check.json'); args = p.parse_args(); sys.exit(run(args))
