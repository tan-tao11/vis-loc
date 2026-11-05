"""Shared inference/training pipeline; PIL rotate then crop then OpenCV resize."""
import math
import cv2
import numpy as np
from PIL import Image
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio import Affine
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union

def center_crop_box(width, height):
    side = min(width, height)
    # Mirrors snapshot integer centre / half-size, including odd dimensions.
    return (width//2-side//2, height//2-side//2, width//2+side//2, height//2+side//2)

def intensity_u8(a, spec, valid=None):
    if a.ndim != 2: raise ValueError('Expected one-channel IR/satellite')
    kind = 'uint8' if a.dtype == np.uint8 else 'uint16' if a.dtype == np.uint16 else 'float' if np.issubdtype(a.dtype, np.floating) else None
    if kind is None: raise ValueError(f'Unsupported intensity dtype {a.dtype}')
    limits = spec.get(kind)
    if limits is None: raise ValueError(f'Explicit {kind} intensity range required')
    lo, hi = limits
    if not np.isfinite([lo, hi]).all() or hi <= lo: raise ValueError('Invalid intensity range')
    mask = np.isfinite(a) if valid is None else valid & np.isfinite(a)
    if spec.get('nodata') is not None: mask &= a != spec['nodata']
    out = np.rint(np.clip((a.astype(np.float32)-lo)/(hi-lo), 0, 1)*255)
    out[~mask] = 0
    return out.astype(np.uint8), mask

def query_image(filename, cfg):
    a = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None: raise ValueError(f'Cannot read {filename}')
    if a.ndim == 3:
        if a.shape[2] < 3 or not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise ValueError('IR image has unequal color channels')
        a = a[..., 0]
    u, valid = intensity_u8(a, cfg['intensity']['ir'])
    box = center_crop_box(a.shape[1], a.shape[0]); l, t, r, b = box
    if not valid[t:b, l:r].all(): raise ValueError('Query crop contains NoData; conservative rejection')
    out = Image.fromarray(u).crop(box).resize((cfg['size'], cfg['size']), Image.Resampling.BICUBIC)
    return np.asarray(out), box

def tensor_image(u8, cfg):
    import torch
    rgb = np.repeat(u8[..., None], 3, axis=-1).astype(np.float32)/255.
    rgb = (rgb-np.array(cfg['mean'], np.float32))/np.array(cfg['std'], np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())

def rotation_affine(width, height, angle):
    """Return PIL-compatible corner-coordinate inverse affine and expanded size."""
    a = -math.radians(angle % 360)
    c, s = round(math.cos(a), 15), round(math.sin(a), 15)
    M = np.array([[c, s, 0.], [-s, c, 0.], [0, 0, 1.]])
    center = np.array([width/2, height/2])
    M[:2, 2] = center-M[:2, :2]@center
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]]) @ M.T
    nw, nh = (np.ceil(corners[:, :2].max(0))-np.floor(corners[:, :2].min(0))).astype(int)
    M[:2, 2] += M[:2, :2] @ np.array([-(nw-width)/2, -(nh-height)/2])
    return M, (int(nw), int(nh))

def local_rotated_crop(raster, inverse, box, spec, max_pixels=16000000):
    """Equivalent to crop(PIL.rotate(full native-resolution uint8 map)).

    A bounded source window includes the bicubic support. Geometry uses pixel
    corners; PIL performs half-pixel sampling internally. No DSM resampling.
    """
    l, t, r, b = map(int, box); w, h = r-l, b-t
    if min(w, h) <= 0 or w*h > max_pixels: raise ValueError('Candidate pixel budget exceeded')
    corners = np.array([[l, t, 1], [r, t, 1], [r, b, 1], [l, b, 1]]) @ inverse.T
    x0, y0 = np.floor(corners[:, :2].min(0)-4).astype(int)
    x1, y1 = np.ceil(corners[:, :2].max(0)+4).astype(int)
    # Clip, rather than pad: PIL's exact edge extension must match full-map rotate.
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(raster.width, x1), min(raster.height, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), np.uint8), np.zeros((h, w), bool), corners[:, :2]
    if (x1-x0)*(y1-y0) > max_pixels: raise ValueError('Source window pixel budget exceeded')
    a = raster.ds.read(1, window=Window(x0, y0, x1-x0, y1-y0), masked=True)
    u, valid = intensity_u8(np.asarray(a), spec, ~np.ma.getmaskarray(a))
    M = inverse.copy(); M[:2, 2] += M[:2, :2] @ [l, t] - [x0, y0]
    coeff = tuple(M[:2].ravel())
    out = np.array(Image.fromarray(u).transform((w, h), Image.Transform.AFFINE, coeff, Image.Resampling.BICUBIC))
    # Require whole interpolation support valid; prevents NoData counting as coverage.
    eroded = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.array(Image.fromarray(eroded).transform((w, h), Image.Transform.AFFINE, coeff, Image.Resampling.NEAREST)) > 0
    return out, mask, corners[:, :2]

def candidate_image(raster, candidate, cfg, max_pixels):
    a, mask, _ = local_rotated_crop(raster, np.array(candidate['inverse']), candidate['box'], cfg['intensity']['satellite'], max_pixels)
    return cv2.resize(a, (cfg['size'], cfg['size']), interpolation=cv2.INTER_LINEAR), mask

def valid_polygon(mask, inverse, box, raster_transform):
    shift = Affine.translation(box[0], box[1])
    rot = Affine(*np.asarray(inverse)[:2].ravel())
    tf = raster_transform * rot * shift
    polygons = [shape(g) for g, v in shapes(mask.astype('uint8'), mask=mask, transform=tf) if v == 1]
    return unary_union(polygons) if polygons else Polygon()
