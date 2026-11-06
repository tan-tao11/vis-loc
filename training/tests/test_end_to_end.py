"""Synthetic end-to-end: prepare -> extract -> train -> export -> inference HeadApplier.

Real training code everywhere except DINO, replaced by a fixed random projection (no weights here).
"""
import importlib.util
import numpy as np
import pytest
import torch
import cache as cache_module
import extract_features as extraction
import models
from common import config, ROOT
from export import export
from prepare_dataset import prepare
from synthetic import make_fixture
from train import train
from evaluate import evaluate_arrays, load_heads_checkpoint

INFERENCE = ROOT.parent/'inference'
DEPLOYED = 'd5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428'


def fake_backbone(monkeypatch):
    g = torch.Generator().manual_seed(0); proj = torch.randn(3*16*16, 1024, generator=g)
    identity = {'name': 'fake_dinov2', 'checkpoint_sha256': DEPLOYED, 'source_tree_sha256': 'fake'}
    def forward(backbone, x, amp):
        f = torch.nn.functional.adaptive_avg_pool2d(x.float(), 16).flatten(1) @ proj.to(x.device)
        return torch.nn.functional.normalize(f, dim=-1)
    for module in (extraction, models):
        monkeypatch.setattr(module, 'backbone_identity', lambda c: identity, raising=False)
    monkeypatch.setattr(extraction, 'load_backbone', lambda c, device: (None, identity))
    monkeypatch.setattr(extraction, 'forward_frozen', forward)
    monkeypatch.setattr(models, 'load_backbone', lambda c, device: (None, identity))


@pytest.mark.skipif(not INFERENCE.is_dir(), reason='inference/ sources not present')
def test_train_export_and_load_in_inference(tmp_path, monkeypatch):
    fake_backbone(monkeypatch)
    c = config(make_fixture(tmp_path/'data'))
    c['output'] = str(tmp_path/'run'); c['train'].update(epochs=3, patience=3, device='cpu')
    m = prepare(c)
    manifest = tmp_path/'run/manifest.json'; features = tmp_path/'run/features'
    extraction.extract(c, manifest, features, 'cpu'); extraction.merge(c, manifest, features, 1)
    train(c, manifest, features, tmp_path/'run/training', 'cpu')
    heads, ckpt = load_heads_checkpoint(tmp_path/'run/training/best.pth')
    _, arrays, _ = cache_module.load_cache(manifest, features)
    report = evaluate_arrays(m, arrays, heads, 'val', 'cpu')
    assert {'gem1024', 'heads'} <= set(report['results'])
    payload = export(tmp_path/'run/training/best.pth', tmp_path/'run/scc_retrieval_head.pth')
    assert payload['backbone']['checkpoint_sha256'] == DEPLOYED and payload['synthetic_only'] is True
    spec = importlib.util.spec_from_file_location('inf_head', INFERENCE/'single-so/python/retrieval_head.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match='synthetic'): mod.HeadApplier(tmp_path/'run/scc_retrieval_head.pth', 'cpu')
    real = torch.load(tmp_path/'run/scc_retrieval_head.pth', weights_only=True); real['synthetic_only'] = False
    torch.save(real, tmp_path/'deploy.pth')                     # what a real-data export looks like
    applier = mod.HeadApplier(tmp_path/'deploy.pth', 'cpu')
    g = next(g for g in m['groups'] if g['split'] == 'val'); q, s = arrays[g['id']]
    q = torch.from_numpy(q)[None]; s = torch.from_numpy(s)
    assert torch.allclose(applier(q, 'ir'), heads(q, 'ir'), atol=1e-6)
    assert torch.allclose(applier(s, 'satellite'), heads(s, 'satellite'), atol=1e-6)
