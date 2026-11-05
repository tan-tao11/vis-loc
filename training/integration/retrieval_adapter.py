import warnings
import torch
from models import RetrievalHeads

class RetrievalAdapter:
    def __init__(self, filename=None, enabled=False, device='cpu', expected_backbone_hash=None,
                 expected_preprocess_pose=None, allow_fallback=False, allow_synthetic=False):
        self.model = None; self.metadata = None
        if not enabled: return
        try:
            if expected_backbone_hash is None or expected_preprocess_pose is None:
                raise ValueError('Explicit expected backbone/preprocessing fingerprints required')
            payload = torch.load(filename, map_location='cpu', weights_only=True)
            if payload['format_version'] != 1: raise ValueError('Unsupported exported head version')
            if payload['synthetic_only'] and not allow_synthetic: raise ValueError('Synthetic diagnostic weights forbidden in production')
            if payload['backbone']['checkpoint_sha256'] != expected_backbone_hash: raise ValueError('Backbone identity mismatch')
            if payload['preprocess_pose_fingerprint'] != expected_preprocess_pose: raise ValueError('Preprocessing/pose mismatch')
            self.model = RetrievalHeads(**payload['structure'])
            self.model.load_state_dict(payload['state_dict'], strict=True); self.model.to(device).eval()
            self.model.requires_grad_(False); self.metadata = payload
        except Exception as e:
            if not allow_fallback: raise
            warnings.warn(f'EXPLICIT retrieval head fallback to GeM1024: {e}', RuntimeWarning)
            self.model = None

    @torch.no_grad()
    def apply(self, global_feature, domain):
        if domain not in ('ir', 'satellite'): raise ValueError('domain must be ir/satellite')
        return global_feature if self.model is None else self.model(global_feature.float(), domain)

    def apply_dense_result(self, result, domain):
        global_feature, patch_tokens, cls_token = result
        # Preserve exact object identity, dtype, shape and values of SGVA inputs.
        return self.apply(global_feature, domain), patch_tokens, cls_token
