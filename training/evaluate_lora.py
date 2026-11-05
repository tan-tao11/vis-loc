"""Online full-candidate LoRA evaluation, frozen GeM and optional existing MLP."""
import argparse
from contextlib import nullcontext
import time
import numpy as np
import torch
from common import write_json, digest
from evaluate import evaluate_arrays, load_heads_checkpoint
from lora_data import load_manifest, OnlineImages, encode_group


@torch.no_grad()
def collect_online(model, images, manifest, split, device, batch_size=4, amp=True, disabled=False):
    selected = [g for g in manifest['groups'] if g['split'] == split]
    if not selected: raise ValueError('Empty '+split+' split')
    arrays = {}; was_training = model.training; model.eval()
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        with model.without_lora() if disabled else nullcontext():
            for g in selected:
                try: q, s = encode_group(model, images, g, device, batch_size, amp)
                except (ValueError, OSError, RuntimeError) as exc:
                    raise RuntimeError(f"Online features failed: split={split}, query={g['id']}, candidates={len(g['candidates'])}, batch={batch_size}: {exc}") from exc
                if not torch.isfinite(q).all() or not torch.isfinite(s).all():
                    raise ValueError('Nonfinite online features: '+g['id'])
                arrays[g['id']] = (q.float().cpu().numpy(), s.float().cpu().numpy())
        if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
    finally:
        model.train(was_training)
    return arrays, time.perf_counter()-start


def evaluate_online(model, images, manifest, split, device, batch_size=4, amp=True,
                    compare_baseline=True, mlp_checkpoint=None, identity=None):
    arrays, elapsed = collect_online(model, images, manifest, split, device, batch_size, amp)
    report = evaluate_arrays(manifest, arrays, split=split)
    report['results']['lora_gem1024'] = report['results'].pop('gem1024')
    report['results']['lora_gem1024']['online_feature_seconds'] = elapsed
    report['results']['lora_gem1024']['feature_timing_scope'] = 'image/window I/O, preprocessing and DINO; excludes model loading'
    if compare_baseline or mlp_checkpoint:
        baseline, seconds = collect_online(model, images, manifest, split, device, batch_size, amp, disabled=True)
        heads = None
        if mlp_checkpoint:
            heads, ckpt = load_heads_checkpoint(mlp_checkpoint)
            if ckpt.get('manifest_id') != manifest['manifest_id']:
                raise ValueError('MLP comparison requires its training best.pth from the EXACT same manifest')
            if ckpt['backbone'] != identity: raise ValueError('MLP/LoRA base backbone differs')
        original = evaluate_arrays(manifest, baseline, heads=heads, split=split)
        report['results']['gem1024'] = original['results']['gem1024']
        report['results']['gem1024']['online_feature_seconds'] = seconds
        if heads: report['results']['mlp512'] = original['results']['heads']
    report['evaluation_protocol'] = {
        'all_candidates': True, 'candidate_subsampling': False, 'frozen_feature_cache_used': False,
        'same_manifest_for_all_methods': True, 'amp': amp, 'image_batch_size': batch_size}
    return report


if __name__ == '__main__':
    from export_lora import load_lora_for_manifest
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val'); p.add_argument('--device', default='cuda:0')
    p.add_argument('--image-batch-size', type=int, default=4); p.add_argument('--no-amp', action='store_true')
    p.add_argument('--mlp-checkpoint'); p.add_argument('--output', required=True)
    a = p.parse_args()
    from outputs import write_evaluation, evaluation_attempt
    with evaluation_attempt(a.output):
        m = load_manifest(a.manifest)
        model, ckpt = load_lora_for_manifest(a.checkpoint, m, a.device)
        with OnlineImages(m) as images:
            r = evaluate_online(model, images, m, a.split, a.device, a.image_batch_size, not a.no_amp,
                                mlp_checkpoint=a.mlp_checkpoint, identity=ckpt['backbone'])
        r['adapter_file'] = a.checkpoint
        print('Report:', write_evaluation(a.output, r, m, render_images=True)/'index.html')
    print({k: v['overall'] for k, v in r['results'].items()})
