import copy
import random
import numpy as np
import pytest
import torch
from torch import nn
from lora import LoRAQKV, LoRARetriever
from lora_data import sample_candidates, balanced_schedule
from integration.lora_adapter import LoRAInferenceAdapter
from common import config, ROOT


def spec(checkpointing=False):
    return {'last_n_blocks': 1, 'rank': 2, 'alpha': 4., 'dropout': 0., 'targets': ['q', 'v'], 'gradient_checkpointing': checkpointing}


def test_zero_qv_only_and_disabled_restore():
    torch.manual_seed(1); layer = LoRAQKV(nn.Linear(8, 24), rank=2, alpha=4, dropout=0)
    x = torch.randn(2, 3, 8); original = layer.base(x)
    assert torch.equal(layer(x), original)
    with torch.no_grad():
        layer.lora_B['q'].fill_(.1); layer.lora_B['v'].fill_(.2)
    output = layer(x)
    assert torch.equal(output[..., 8:16], original[..., 8:16])
    assert not torch.equal(output[..., :8], original[..., :8])
    expected_q = original[..., :8]+2*(x @ layer.lora_A['q'].T @ layer.lora_B['q'].T)
    assert torch.allclose(output[..., :8], expected_q)
    layer.enabled = False; assert torch.equal(layer(x), original)
    assert all(not p.requires_grad for p in layer.base.parameters())


def test_lora_learns_without_base_updates():
    torch.manual_seed(4); torch.set_num_threads(2)
    layer = LoRAQKV(nn.Linear(12, 36), rank=3, alpha=3, dropout=0)
    original_state = copy.deepcopy(layer.base.state_dict()); x = torch.randn(32, 12)
    desired_delta = torch.randn(12, 2) @ torch.randn(2, 12)*.1
    target = layer.base(x).detach(); target[:, :12] += x @ desired_delta
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.03)
    history = []
    for _ in range(60):
        opt.zero_grad(); loss = (layer(x)-target).square().mean(); loss.backward(); opt.step(); history.append(loss.item())
    assert history[-1] < history[0]*.1
    assert layer.lora_A['q'].grad.norm() > 0 and layer.lora_B['q'].grad.norm() > 0
    assert all(torch.equal(v, original_state[k]) for k, v in layer.base.state_dict().items())
    assert all(p.grad is None for p in layer.base.parameters())


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = nn.Module(); self.attn.qkv = nn.Linear(1024, 3072)
    def forward(self, x):
        q, k, v = self.attn.qkv(x).chunk(3, -1)
        return x+(q+.2*k+v).tanh()*.2


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()]); self.norm = nn.LayerNorm(1024)
    def prepare_tokens_with_masks(self, x): return x


def test_checkpoint_gradients_and_state_reload():
    torch.manual_seed(9); base = TinyDino(); other = copy.deepcopy(base)
    torch.manual_seed(3); direct = LoRARetriever(base, spec(False))
    torch.manual_seed(3); checked = LoRARetriever(other, spec(True))
    x = torch.randn(2, 5, 1024)  # frozen-prefix output has no requires_grad
    weight = torch.randn(2, 1024)
    direct.train(); checked.train()
    (direct(x)*weight).sum().backward(); (checked(x)*weight).sum().backward()
    for (name, p), (_, q) in zip(direct.named_parameters(), checked.named_parameters()):
        if p.requires_grad: assert torch.allclose(p.grad, q.grad, atol=1e-6), name
        else: assert p.grad is None and q.grad is None
    assert checked.adapters()[0].lora_B['q'].grad.norm() > 0
    assert checked.assert_freeze() == 8192
    saved = checked.adapter_state(); checked.load_adapter_state(saved)
    assert all('lora_A' in k or 'lora_B' in k for k in saved)
    with pytest.raises(ValueError, match='keys'): checked.load_adapter_state({})
    corrupt = {k: v.clone() for k, v in saved.items()}; next(iter(corrupt.values())).fill_(float('nan'))
    with pytest.raises(ValueError, match='Invalid'): checked.load_adapter_state(corrupt)
    with pytest.raises(RuntimeError):
        with checked.without_lora():
            assert not checked.adapters()[0].enabled
            raise RuntimeError('restore the flag')
    assert checked.adapters()[0].enabled


def test_sampling_excludes_unknown_and_no_val_mining():
    c = config(ROOT/'configs/lora.yaml')['train']
    group = {'id': 'a', 'group': 'DS', 'split': 'train', 'candidates': [{'label': v} for v in [1, 1, -1, 0, 0, 0]]}
    sampled = sample_candidates(group, [3], c, random.Random(1))
    assert 2 not in sampled and set(sampled) >= {0, 1, 3}
    with pytest.raises(ValueError): sample_candidates(group, [2], c, random.Random(1))
    group['split'] = 'val'
    with pytest.raises(ValueError): sample_candidates(group, [3], c, random.Random(1))
    groups = [{'id': str(i), 'group': 'DS' if i < 3 else 'NV'} for i in range(4)]
    schedule = balanced_schedule(groups, random.Random(1))
    assert set(g['id'] for g in schedule) == set(g['id'] for g in groups)
    assert len(schedule) == 6


def test_disabled_integration_keeps_sgva():
    adapter = LoRAInferenceAdapter(enabled=False)
    original = (torch.randn(1, 1024), torch.randn(1, 576, 1024), torch.randn(1, 1024))
    result = adapter.replace_global(original, torch.randn(1, 3, 336, 336))
    assert all(a is b for a, b in zip(original, result))
    with pytest.raises(ValueError): adapter.retrieve(torch.zeros(1))


@pytest.mark.parametrize('kwargs', [{'rank': 0}, {'rank': 9}, {'targets': ['x']}, {'targets': ['q', 'q']}, {'dropout': 1.}])
def test_invalid_lora_configuration(kwargs):
    with pytest.raises(ValueError): LoRAQKV(nn.Linear(8, 24), **kwargs)


def test_training_resume_preserves_optimizer_rng_and_result(tmp_path, monkeypatch):
    import train_lora as trainer
    torch.set_num_threads(2)
    c = config(ROOT/'configs/lora.yaml')
    c['train'].update(epochs=3, patience=3, device='cpu', amp=False, accumulation_groups=2,
                      hard_negatives=1, random_negatives=1, max_positives=1)
    c['lora'] = {**spec(True), 'dropout': .1}
    groups = [{'id': str(i), 'group': 'DS' if i == 0 else 'NV', 'split': 'train' if i < 2 else 'val',
        'ground_center_map': [0, 0], 'candidates': [
            {'label': label, 'center_map': [j, j]} for j, label in enumerate([1, 0, -1, 0])]} for i in range(3)]
    manifest = {'manifest_id': 'resume-fixture', 'synthetic': True, 'groups': groups,
                'config': {'preprocess': {}, 'pose': {}}}
    monkeypatch.setattr(trainer, 'load_manifest', lambda _: manifest)
    monkeypatch.setattr(trainer, 'build_lora', lambda _, setting, device: (LoRARetriever(TinyDino(), setting), {'fixture': True}))
    inputs = torch.randn(3, 5, 5, 1024, generator=torch.Generator().manual_seed(18))
    class Images:
        def __init__(self, _): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, group, index=None): return inputs[int(group['id']), 0 if index is None else index+1]
    monkeypatch.setattr(trainer, 'OnlineImages', Images)
    monkeypatch.setattr(trainer, 'evaluate_online', lambda *args, **kwargs: {
        'results': {'lora_gem1024': {'overall': {
            'recall_all_legal': {'1': .5, '5': 1.}, 'mean_first_positive_rank_covered': 2.}}}})
    trainer.train_lora(c, 'unused', tmp_path/'full', 'cpu')
    save = trainer.atomic_save
    def interrupt_after_checkpoint(filename, payload):
        save(filename, payload)
        if filename.name == 'last.pth' and payload['epoch'] == 0: raise RuntimeError('simulated interruption')
    monkeypatch.setattr(trainer, 'atomic_save', interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        trainer.train_lora(c, 'unused', tmp_path/'resumed', 'cpu')
    monkeypatch.setattr(trainer, 'atomic_save', save)
    trainer.train_lora(c, 'unused', tmp_path/'resumed', 'cpu', tmp_path/'resumed/last.pth')
    full = torch.load(tmp_path/'full/last.pth', weights_only=True)
    resumed = torch.load(tmp_path/'resumed/last.pth', weights_only=True)
    assert full['history'] == resumed['history']
    assert all(torch.equal(value, resumed['state_dict'][key]) for key, value in full['state_dict'].items())
    assert full['scheduler'] == resumed['scheduler']
