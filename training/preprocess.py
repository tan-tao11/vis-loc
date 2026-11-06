"""Query image preprocessing (centre square crop + PIL BICUBIC, as SCC crop_center + resize).

Satellite candidate tiles are produced by scc_inference.py from the deployed code.
"""
import cv2
import numpy as np
from PIL import Image

def center_crop_box(width, height, scale=1.0):
    side = int(min(width, height)*scale)
    # Mirrors snapshot integer centre / half-size, including odd dimensions.
    return (width//2-side//2, height//2-side//2, width//2+side//2, height//2+side//2)

def intensity_u8(a, spec, valid=None, *, mask_only=False):
    if a.ndim != 2: raise ValueError('Expected one-channel IR/satellite')
    kind = 'uint8' if a.dtype == np.uint8 else 'uint16' if a.dtype == np.uint16 else 'float' if np.issubdtype(a.dtype, np.floating) else None
    if kind is None: raise ValueError(f'Unsupported intensity dtype {a.dtype}')
    limits = spec.get(kind)
    if limits is None: raise ValueError(f'Explicit {kind} intensity range required')
    lo, hi = limits
    if not np.isfinite([lo, hi]).all() or hi <= lo: raise ValueError('Invalid intensity range')
    mask = np.isfinite(a) if valid is None else valid & np.isfinite(a)
    if spec.get('nodata') is not None: mask &= a != spec['nodata']
    if mask_only: return None, mask
    out = np.rint(np.clip((a.astype(np.float32)-lo)/(hi-lo), 0, 1)*255)
    out[~mask] = 0
    return out.astype(np.uint8), mask

def query_gray(filename, cfg):
    """Whole IR frame as uint8 gray (+ valid mask), before any crop."""
    a = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None: raise ValueError(f'Cannot read {filename}')
    if a.ndim == 3:
        if a.shape[2] < 3 or not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError('IR image has unequal color channels')
        a = a[..., 0]
    return intensity_u8(a, cfg['intensity']['ir'])

def query_image(filename, cfg):
    u, valid = query_gray(filename, cfg)
    a = u
    box = center_crop_box(a.shape[1], a.shape[0], float(cfg.get('query_crop_scale', 1.0))); l, t, r, b = box
    if not valid[t:b, l:r].all(): raise ValueError('Query crop contains NoData; conservative rejection')
    out = Image.fromarray(u).crop(box).resize((cfg['size'], cfg['size']), Image.Resampling.BICUBIC)
    return np.asarray(out), box

def tensor_image(u8, cfg):
    import torch
    rgb = np.repeat(u8[..., None], 3, axis=-1).astype(np.float32)/255.
    rgb = (rgb-np.array(cfg['mean'], np.float32))/np.array(cfg['std'], np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


