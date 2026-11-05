"""Generate a narrowly scoped patch against the PROVIDED snapshot, not unknown .so."""
from pathlib import Path
import difflib

root = Path(__file__).resolve().parents[1]
source = (root/'reference/scc_snapshot/feature_extract.py').read_text(encoding='utf-8')
target = source.replace('UAV, val_transforms, model):', 'UAV, val_transforms, model, retrieval_head=None):')
q = '                uav_global_feat, _, uav_cls_token = model(UAV1, return_dense=True)'
target = target.replace(q, q+'\n                if retrieval_head is not None:\n                    uav_global_feat = retrieval_head.apply(uav_global_feat, domain="ir")')
s = '                        sat_global, sat_dense, _ = model(batch_images, return_dense=True)'
target = target.replace(s, s+'\n                        if retrieval_head is not None:\n                            sat_global = retrieval_head.apply(sat_global, domain="satellite")')
assert source != target and target.count('retrieval_head.apply(') == 2
patch = ''.join(difflib.unified_diff(source.splitlines(True), target.splitlines(True),
                fromfile='a/Retrieval_Models/feature_extract.py', tofile='b/Retrieval_Models/feature_extract.py'))
(root/'integration/feature_extract.snapshot.patch').write_text(patch, encoding='utf-8')
print('Generated snapshot-specific patch with 2 global feature applications')
