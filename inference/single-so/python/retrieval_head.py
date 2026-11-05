"""Optional trained MLP retrieval head (training/export.py -> scc_retrieval_head.pth).

RetrievalHeads is a verbatim copy of training/models.py (a training test compares them). The head
only replaces the 1024-d GeM retrieval descriptor: query -> 'ir', satellite blocks -> 'satellite'.
Dense/CLS tokens for SGVA and the RoMa matching path are untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F

DEPLOYED_DINO_SHA256 = 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'


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


class HeadApplier:
    """Callable used by feature_extract: FP32, no autocast, no gradients."""
    def __init__(self, filename, device):
        try:
            payload = torch.load(filename, map_location='cpu', weights_only=True)
        except TypeError:   # older torch without weights_only
            payload = torch.load(filename, map_location='cpu')
        if payload.get('format_version') != 1: raise ValueError('unsupported retrieval head format')
        if payload.get('synthetic_only'): raise ValueError('synthetic diagnostic head refused for deployment')
        if payload['backbone']['checkpoint_sha256'] != DEPLOYED_DINO_SHA256:
            raise ValueError('retrieval head was trained on a different DINOv2 checkpoint')
        gem = payload.get('gem', {})
        if gem.get('p') != 4. or gem.get('dimension') != 1024: raise ValueError('retrieval head expects GeM p=4, 1024-d')
        self.model = RetrievalHeads(**payload['structure'])
        self.model.load_state_dict(payload['state_dict'], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.output_dim = int(payload['structure']['output_dim'])

    def __call__(self, global_feature, domain):
        with torch.no_grad(), torch.autocast(device_type=global_feature.device.type, enabled=False):
            return self.model(global_feature.float().clone(), domain)
