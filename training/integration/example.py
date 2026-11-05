"""Run from project root: python -m integration.example"""
import torch
from integration.retrieval_adapter import RetrievalAdapter

def retrieve(query_result, gallery_result, adapter):
    q, qpatch, qcls = adapter.apply_dense_result(query_result, 'ir')
    sat, dense, cls = adapter.apply_dense_result(gallery_result, 'satellite')
    scores = sat @ q.T
    # SGVA consumes unchanged 1024-D CLS × 1024-D dense features.
    heatmap = torch.nn.functional.normalize(dense, dim=-1) @ torch.nn.functional.normalize(qcls, dim=-1).unsqueeze(-1)
    return scores, heatmap

if __name__ == '__main__':
    q = (torch.randn(1, 1024), torch.randn(1, 576, 1024), torch.randn(1, 1024))
    s = (torch.randn(2, 1024), torch.randn(2, 576, 1024), torch.randn(2, 1024))
    scores, heatmap = retrieve(q, s, RetrievalAdapter(enabled=False))
    assert torch.equal(scores, s[0] @ q[0].T)
    print('Disabled path unchanged; scores', scores.shape, 'SGVA', heatmap.shape)
