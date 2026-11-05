import argparse
from pathlib import Path
import torch
from evaluate import load_heads_checkpoint
from common import digest, sha256, write_json
from outputs import atomic_torch_export

def export(checkpoint, output):
    model, ckpt = load_heads_checkpoint(checkpoint)
    c = ckpt['config']
    # Whitelist inference metadata. No GT, dataset paths, optimizer or backbone copy.
    metadata = {'format_version': 1, 'structure': model.structure, 'gem': {'p': 4., 'clamp_min': 1e-6, 'dimension': 1024},
                'preprocess': c['preprocess'], 'pose': c['pose'], 'backbone': ckpt['backbone'],
                'synthetic_only': ckpt['synthetic'], 'selection_split': ckpt['selection_split'],
                'training_code_version': '1.0', 'state_dict': {k: v.cpu().contiguous() for k, v in model.state_dict().items()}}
    metadata['preprocess_pose_fingerprint'] = digest({'preprocess': metadata['preprocess'], 'pose': metadata['pose']})
    if not all(torch.isfinite(v).all() for v in metadata['state_dict'].values()): raise ValueError('Nonfinite export tensor')
    atomic_torch_export(output, metadata)
    write_json(str(output)+'.json', {k: v for k, v in metadata.items() if k != 'state_dict'} | {'sha256': sha256(output)})
    return metadata

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--output', default='scc_retrieval_head.pth')
    a = p.parse_args(); export(a.checkpoint, a.output)
