import numpy as np
import torch
import pytest
from models import RetrievalHeads, gem
from losses import multi_positive_loss
from evaluate import rank_metrics, summarize
from integration.retrieval_adapter import RetrievalAdapter
from integration.example import retrieve

def test_multi_positive_exact_and_empty():
    z = torch.tensor([[1., 2., 100., 3.]], requires_grad=True)
    p = torch.tensor([[True, True, False, False]]); legal = torch.tensor([[True, True, False, True]])
    loss, count = multi_positive_loss(z, p, legal)
    expected = torch.logsumexp(z[0, [0, 1, 3]], 0)-torch.logsumexp(z[0, :2], 0)
    assert torch.allclose(loss, expected) and count == 1
    loss.backward(); assert z.grad[0, 2] == 0
    loss, count = multi_positive_loss(z, torch.zeros_like(p), legal); assert loss == 0 and count == 0
    with pytest.raises(ValueError): multi_positive_loss(z, p, ~legal)

def test_uncovered_queries_count_as_misses():
    rows = [rank_metrics([.9, .2], [1, 0]), rank_metrics([.9, .2], [0, -1]), rank_metrics([], [])]
    m = summarize(rows); assert m['recall_all_legal']['1'] == 1/3; assert m['recall_conditional_covered']['1'] == 1

def test_both_heads_learn_frozen_backbone(tmp_path):
    torch.manual_seed(7); torch.set_num_threads(2)
    backbone = torch.nn.Linear(32, 1024).requires_grad_(False)
    with torch.no_grad(): q = torch.nn.functional.normalize(backbone(torch.randn(12, 32)), dim=-1).clone()
    sat = q[:, torch.randperm(1024)].clone()
    heads = RetrievalHeads(hidden_dim=64, output_dim=32)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=.003)
    p = torch.eye(12, dtype=torch.bool); legal = torch.ones_like(p); values = []; grads = {}
    for step in range(60):
        optimizer.zero_grad(); z = heads(q, 'ir') @ heads(sat, 'satellite').T / .1
        loss, _ = multi_positive_loss(z, p, legal); loss.backward()
        if step == 0:
            grads = {d: sum(float(v.grad.norm()) for v in getattr(heads, d).parameters()) for d in ('ir', 'satellite')}
        optimizer.step(); values.append(float(loss.detach()))
    assert all(x > 0 for x in grads.values()); assert all(p.grad is None for p in backbone.parameters())
    assert values[-1] < values[0]*.2
    file = tmp_path/'state.pth'; torch.save(heads.state_dict(), file)
    again = RetrievalHeads(hidden_dim=64, output_dim=32); again.load_state_dict(torch.load(file, weights_only=True))
    assert torch.equal(heads(q, 'ir'), again(q, 'ir'))

def test_gem_and_disabled_sgva():
    tokens = torch.randn(2, 576, 1024)
    expected = torch.nn.functional.normalize(tokens.clamp(min=1e-6).pow(4).mean(1).pow(.25), dim=-1)
    assert torch.equal(gem(tokens), expected)
    a = RetrievalAdapter(enabled=False)
    q = (torch.randn(1, 1024), torch.randn(1, 576, 1024), torch.randn(1, 1024))
    s = (torch.randn(2, 1024), tokens, torch.randn(2, 1024))
    result = a.apply_dense_result(s, 'satellite'); assert all(x is y for x, y in zip(s, result))
    scores, heat = retrieve(q, s, a); assert torch.equal(scores, s[0] @ q[0].T); assert heat.shape == (2, 576, 1)
