"""Independent retrieval DINO; original SCC dense/CLS/matching tensors are preserved."""
import warnings
import torch
from common import digest
from lora import build_lora
from export_lora import preprocessing_fingerprint


class LoRAInferenceAdapter:
    def __init__(self, filename=None, data_config=None, enabled=False, device='cpu',
                 allow_fallback=False, allow_synthetic=False):
        self.model = None; self.metadata = None; self.device = device
        if not enabled: return
        try:
            if data_config is None: raise ValueError('Deployment config with local source/checkpoint and preprocessing is required')
            payload = torch.load(filename, weights_only=True, map_location='cpu')
            if payload.get('kind') != 'scc_dinov2_lora' or payload['format_version'] != 1:
                raise ValueError('Expected exported LoRA artifact, not MLP or training checkpoint')
            if payload['synthetic_only'] and not allow_synthetic: raise ValueError('Synthetic LoRA forbidden in production')
            fp = preprocessing_fingerprint(data_config['preprocess'], data_config['pose'])
            if fp != payload['preprocess_pose_fingerprint']: raise ValueError('LoRA preprocessing/pose mismatch')
            # Fresh local load guarantees no LoRA is injected/merged into SCC's shared matching backbone.
            model, identity = build_lora(data_config, payload['lora'], device)
            if payload['backbone'] != identity: raise ValueError('LoRA original backbone/source hash mismatch')
            model.load_adapter_state(payload['state_dict']); model.eval()
            self.model, self.metadata = model, payload
        except Exception as e:
            if not allow_fallback: raise
            warnings.warn(f'EXPLICIT LoRA fallback to ORIGINAL GeM1024: {e}', RuntimeWarning)

    @torch.no_grad()
    def retrieve(self, normalized_images, original_global=None, amp=True):
        if self.model is None:
            if original_global is None: raise ValueError('Disabled/fallback LoRA requires original global features')
            return original_global
        if normalized_images.ndim != 4 or normalized_images.shape[1:] != (3, 336, 336):
            raise ValueError('Pass the SAME normalized 3x336x336 image tensor used by SCC retrieval')
        enabled = amp and torch.device(self.device).type == 'cuda'
        with torch.autocast(torch.device(self.device).type, dtype=torch.float16, enabled=enabled):
            return self.model(normalized_images.to(self.device))

    def replace_global(self, original_result, normalized_images, amp=True):
        original_global, original_dense, original_cls = original_result
        if original_dense.shape[-1] != 1024 or original_cls.shape[-1] != 1024:
            raise ValueError('SGVA must retain original 1024-dimensional DINO dense and CLS')
        adapted = self.retrieve(normalized_images, original_global, amp)
        return adapted, original_dense, original_cls
