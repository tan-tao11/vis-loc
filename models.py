import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from common import path, sha256, digest

def gem(tokens, p=4.):
    if p != 4.: raise ValueError('Deployed GeM exponent must be fixed at 4')
    if tokens.shape[-1] != 1024: raise ValueError('Expected DINOv2 ViT-L/14 1024-dimensional tokens')
    x = tokens.float().clamp(min=1e-6).pow(4).mean(1).pow(.25)
    if not torch.isfinite(x).all(): raise ValueError('Nonfinite GeM features')
    return F.normalize(x, dim=-1)

class RetrievalHeads(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=512, dropout=0.):
        super().__init__()
        if input_dim != 1024: raise ValueError('Backbone dimension must remain 1024')
        self.structure = dict(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
        def head(): return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
        self.ir = head(); self.satellite = head()

    def forward(self, x, domain):
        if domain not in ('ir', 'satellite'): raise ValueError('Explicit domain ir/satellite required')
        return F.normalize(getattr(self, domain)(x.float()), dim=-1)

def create_heads(c):
    return RetrievalHeads(**{k: c['model'][k] for k in ('input_dim', 'hidden_dim', 'output_dim', 'dropout')})

def backbone_identity(c):
    m = c['model']; src = path(m['source']); ckpt = path(m['checkpoint'])
    if m['name'] != 'dinov2_vitl14': raise ValueError('Only non-register DINOv2 ViT-L/14 supported')
    if not (src/'hubconf.py').is_file() or not ckpt.is_file(): raise FileNotFoundError('Local DINO source/checkpoint missing')
    weights_hash = sha256(ckpt)
    if m['checkpoint_sha256'] and weights_hash != m['checkpoint_sha256']: raise ValueError('Backbone checkpoint hash mismatch')
    source_hash = digest({str(p.relative_to(src)): sha256(p) for p in sorted(src.rglob('*.py'))})
    commit_file = src/'SOURCE_COMMIT.txt'
    commit = commit_file.read_text().strip() if commit_file.exists() else None
    if m['source_commit'] and m['source_commit'] != commit: raise ValueError('DINO source commit mismatch')
    return {'name': m['name'], 'checkpoint_sha256': weights_hash, 'source_tree_sha256': source_hash,
            'source_commit': commit, 'deployed_backbone_confirmed': m['deployed_backbone_confirmed']}

def load_backbone(c, device='cpu'):
    identity = backbone_identity(c)
    os.environ['XFORMERS_DISABLED'] = '1'
    backbone = torch.hub.load(str(path(c['model']['source'])), c['model']['name'], source='local', pretrained=False)
    state = torch.load(path(c['model']['checkpoint']), map_location='cpu', weights_only=True)
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(False); backbone.eval(); backbone.to(device)
    return backbone, identity

def forward_frozen(backbone, batch, amp=True):
    with torch.no_grad(), torch.autocast(device_type=batch.device.type, dtype=torch.float16,
                                         enabled=amp and batch.device.type == 'cuda'):
        result = backbone.forward_features(batch)
        tokens = result['x_norm_patchtokens']
    # no inference_mode tensors cross into trainable heads; GeM always float32.
    return gem(tokens).detach().clone()
