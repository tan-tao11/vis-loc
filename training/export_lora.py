"""Portable adapter-only artifact; never merge into SCC's matching backbone."""
import argparse
from pathlib import Path
import torch
from common import digest, sha256, write_json
from lora import build_lora
from outputs import atomic_torch_export


def preprocessing_fingerprint(preprocess, pose):
    return digest({'preprocess': preprocess, 'pose': pose})


def export_lora(checkpoint, output):
    trained = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if trained.get('kind') != 'scc_dinov2_lora_training' or trained['selection_split'] != 'val':
        raise ValueError('Expected validation-selected LoRA training checkpoint')
    keep = ('format_version', 'lora', 'backbone', 'preprocess', 'pose', 'gem', 'output_dim', 'synthetic_only', 'selection_split')
    payload = {k: trained[k] for k in keep}
    payload['kind'] = 'scc_dinov2_lora'
    payload['preprocess_pose_fingerprint'] = preprocessing_fingerprint(payload['preprocess'], payload['pose'])
    payload['state_dict'] = {}
    for name, value in trained['state_dict'].items():
        if '.lora_A.' not in name and '.lora_B.' not in name: raise ValueError('Export refuses original backbone weights')
        if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all(): raise ValueError('Invalid adapter tensor')
        payload['state_dict'][name] = value.detach().float().cpu().contiguous()
    if not payload['state_dict']: raise ValueError('Empty adapter')
    dest = Path(output)
    atomic_torch_export(dest, payload)
    write_json(str(dest)+'.json', {k: v for k, v in payload.items() if k != 'state_dict'} | {'sha256': sha256(dest)})
    return payload


def load_lora_for_manifest(filename, manifest, device='cpu'):
    ckpt = torch.load(filename, map_location='cpu', weights_only=True)
    if ckpt.get('kind') not in ('scc_dinov2_lora', 'scc_dinov2_lora_training') or ckpt['format_version'] != 1:
        raise ValueError('Unknown LoRA checkpoint')
    if ckpt.get('manifest_id', manifest['manifest_id']) != manifest['manifest_id']: raise ValueError('Training checkpoint/manifest mismatch')
    c = manifest['config']
    if preprocessing_fingerprint(ckpt['preprocess'], ckpt['pose']) != preprocessing_fingerprint(c['preprocess'], c['pose']):
        raise ValueError('LoRA preprocessing/pose mismatch')
    if ckpt['synthetic_only'] and not manifest['synthetic']: raise ValueError('Synthetic LoRA forbidden on real data')
    model, identity = build_lora(c, ckpt['lora'], device)
    if ckpt['backbone'] != identity: raise ValueError('LoRA base checkpoint/source identity mismatch')
    model.load_adapter_state(ckpt['state_dict']); model.eval()
    return model, ckpt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--output', default='scc_retrieval_lora.pth')
    a = p.parse_args(); export_lora(a.checkpoint, a.output)
