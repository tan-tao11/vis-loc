"""Generate optional LoRA hooks against the supplied SCC source snapshot only."""
from pathlib import Path
import ast
import difflib

root = Path(__file__).resolve().parents[1]
source = (root/'reference/scc_snapshot/feature_extract.py').read_text(encoding='utf-8')
signature = 'UAV, val_transforms, model):'
q = '                uav_global_feat, _, uav_cls_token = model(UAV1, return_dense=True)'
s = '                        sat_global, sat_dense, _ = model(batch_images, return_dense=True)'
for anchor in (signature, q, s):
    if source.count(anchor) != 1: raise ValueError('Snapshot changed: expected one exact anchor')
target = source.replace(signature, 'UAV, val_transforms, model, retrieval_lora=None):')
target = target.replace(q, q+'\n                if retrieval_lora is not None:\n                    uav_global_feat = retrieval_lora.retrieve(UAV1, uav_global_feat)')
target = target.replace(s, s+'\n                        if retrieval_lora is not None:\n                            sat_global = retrieval_lora.retrieve(batch_images, sat_global)')
ast.parse(target)
patch = ''.join(difflib.unified_diff(source.splitlines(True), target.splitlines(True),
    fromfile='a/Retrieval_Models/feature_extract.py', tofile='b/Retrieval_Models/feature_extract.py'))
(root/'integration/feature_extract.lora.snapshot.patch').write_text(patch, encoding='utf-8')
print('Generated LoRA snapshot patch; original dense/CLS and matching model retained')
