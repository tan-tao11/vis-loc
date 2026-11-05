"""Native PyTorch LoRA on fused DINO QKV. No PEFT, compiler or CUDA extension."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models import gem, load_backbone


class LoRAQKV(nn.Module):
    def __init__(self, base, rank=8, alpha=16., dropout=.05, targets=('q', 'v')):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features != base.in_features*3:
            raise ValueError('Expected original fused qkv Linear(D, 3D)')
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= base.in_features:
            raise ValueError('LoRA rank must be an integer in [1, D]')
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid alpha/dropout')
        if not targets or len(set(targets)) != len(targets) or set(targets)-{'q', 'k', 'v'}:
            raise ValueError('LoRA targets must be unique q/k/v slices')
        self.base = base.requires_grad_(False)
        self.rank, self.scale = rank, float(alpha)/rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in targets:
            self.lora_A[name] = nn.Parameter(base.weight.new_empty(rank, base.in_features))
            self.lora_B[name] = nn.Parameter(base.weight.new_zeros(base.in_features, rank))
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        dropped = self.dropout(x)
        pieces = []
        for name, part in zip(('q', 'k', 'v'), original.chunk(3, dim=-1)):
            if name in self.lora_A:
                delta = F.linear(F.linear(dropped, self.lora_A[name]), self.lora_B[name])
                part = part + delta*self.scale
            pieces.append(part)
        return torch.cat(pieces, dim=-1)


class LoRARetriever(nn.Module):
    """Owns a retrieval-only backbone. No mutation of a passed SCC matching model."""
    def __init__(self, backbone, spec):
        super().__init__()
        if getattr(backbone, 'chunked_blocks', False) or getattr(backbone, 'num_register_tokens', 0) != 0:
            raise ValueError('Only non-chunked, non-register DINOv2 is supported')
        n = spec['last_n_blocks']
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(backbone.blocks):
            raise ValueError('last_n_blocks must be in [1, number of blocks]')
        self.backbone = backbone.requires_grad_(False)
        self.spec = dict(spec)
        self.block_indices = list(range(len(backbone.blocks)-n, len(backbone.blocks)))
        for i in self.block_indices:
            attention = backbone.blocks[i].attn
            attention.qkv = LoRAQKV(attention.qkv, spec['rank'], spec['alpha'], spec['dropout'], spec['targets'])
        self.train(False)

    def adapters(self):
        return [self.backbone.blocks[i].attn.qkv for i in self.block_indices]

    def train(self, mode=True):
        super().train(mode)
        # Freeze stochastic depth and original DINO dropout; LoRA alone may use dropout.
        self.backbone.eval()
        for adapter in self.adapters():
            adapter.dropout.train(mode)
        return self

    @contextmanager
    def without_lora(self):
        adapters = self.adapters(); previous = [a.enabled for a in adapters]
        for a in adapters: a.enabled = False
        try: yield
        finally:
            for a, enabled in zip(adapters, previous): a.enabled = enabled

    def forward(self, images):
        # Frozen prefix can run without graph; non-reentrant checkpoint retains
        # LoRA parameter gradients even though this prefix output requires no grad.
        with torch.no_grad():
            x = self.backbone.prepare_tokens_with_masks(images)
            for block in self.backbone.blocks[:self.block_indices[0]]:
                x = block(x)
        for block in self.backbone.blocks[self.block_indices[0]:]:
            if self.training and torch.is_grad_enabled() and self.spec['gradient_checkpointing']:
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = block(x)
        normalized = self.backbone.norm(x)
        return gem(normalized[:, 1:])  # GeM in FP32, unchanged p=4; no additional MLP.

    def adapter_state(self):
        return {name: value.detach().cpu().contiguous().clone() for name, value in self.named_parameters()
                if value.requires_grad}

    def load_adapter_state(self, state):
        expected = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if set(state) != set(expected): raise ValueError('LoRA state keys do not match injection configuration')
        for name, parameter in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid LoRA tensor: '+name)
        with torch.no_grad():
            for name, parameter in expected.items(): parameter.copy_(state[name])

    def assert_freeze(self):
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        if not trainable or any('.lora_A.' not in name and '.lora_B.' not in name for name in trainable):
            raise RuntimeError('Only LoRA parameters may be trainable')
        if any(p.grad is not None for p in self.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen base parameter received a gradient')
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lora(data_config, lora_spec, device='cpu'):
    # Always create a fresh local instance. Never inject adapters into MINIMA-RoMa.
    backbone, identity = load_backbone(data_config, device)
    if len(backbone.blocks) != 24 or backbone.embed_dim != 1024:
        raise ValueError('Expected DINOv2 ViT-L/14: 24 blocks and dimension 1024')
    model = LoRARetriever(backbone, lora_spec).to(device)
    model.assert_freeze()
    return model, identity
