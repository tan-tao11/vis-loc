import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/SCC-Loc/Retrieval_Models'))
import feature_extract


class FakeDino(torch.nn.Module):
    def forward(self, x, return_dense=False):
        g = torch.nn.functional.normalize(x.mean(dim=(2, 3)).repeat(1, 342)[:, :1024], dim=-1)
        dense = torch.ones(x.shape[0], 576, 1024); cls = torch.nn.functional.normalize(torch.ones(x.shape[0], 1024), dim=-1)
        return g, dense, cls


class RetrievalHeadIntegrationTests(unittest.TestCase):
    def run_extract(self, head):
        torch.manual_seed(0)
        sat = (torch.rand(400, 400, 3)*255).byte().numpy()
        uav = torch.rand(3, 336, 336)
        mids = torch.tensor([[100., 100.], [200., 250.], [300., 300.]]).numpy()
        cfg = {'DEVICE': 'cpu', 'BATCH_SIZE': 2, 'RETRIEVAL_FEATURE_NORM': True}
        method = {'retrieval_method': 'MINIMA_Roma_DINOv2', 'retrieval_head': head}
        tf = lambda img: torch.from_numpy(__import__('numpy').asarray(img)).permute(2, 0, 1).float()/255
        return feature_extract.extract_features(336, cfg, method, mids, [100, 100], sat, uav, tf, FakeDino())

    def test_head_applied_by_domain_and_absent_head_unchanged(self):
        calls = []
        def head(x, domain):
            calls.append((domain, tuple(x.shape))); return torch.nn.functional.normalize(x[:, :512]*(2 if domain == 'ir' else 3), dim=-1)
        gf0, qf0, _ = self.run_extract(None)
        self.assertEqual(tuple(gf0.shape), (3, 1024)); self.assertEqual(tuple(qf0.shape), (1, 1024))
        gf, qf, _ = self.run_extract(head)
        self.assertEqual(tuple(gf.shape), (3, 512)); self.assertEqual(tuple(qf.shape), (1, 512))
        self.assertEqual(calls[0], ('ir', (1, 1024)))
        self.assertTrue(all(d == 'satellite' for d, _ in calls[1:])); self.assertEqual(sum(s[0] for _, s in calls[1:]), 3)
        self.assertEqual(tuple((gf @ qf.unsqueeze(-1)).squeeze().shape), (3,))   # retrieval_all scoring still works


if __name__ == '__main__':
    unittest.main()
