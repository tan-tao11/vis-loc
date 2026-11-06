"""Oblique (slant) helpers shared VERBATIM by inference and training (training/scc_oblique.py is a byte copy).

Conventions: the SCC search map is rotated by the camera heading `yaw` (dumpRotateImage), so the
camera looks UP in the rotated map. `elevation` is the optical-axis elevation in degrees (negative =
below the horizon). Ground is taken as a flat plane `agl` metres below the camera; image roll is
ignored (the competition slant data has |roll| < 1.5 deg).
"""
import math
import cv2
import numpy as np


def view_offset(agl, elevation):
    """Horizontal distance (m) from the aircraft to the ground point on the optical axis."""
    a = math.radians(-float(elevation))
    if not 0 < a <= math.pi / 2 or not agl > 0:
        raise ValueError('OBLIQUE: invalid elevation or AGL')
    return 0.0 if a >= math.pi / 2 - 1e-9 else float(agl) / math.tan(a)


def view_centre(x, y, agl, yaw, elevation):
    """Working-CRS metres of the optical-axis ground point (yaw clockwise from grid north)."""
    d = view_offset(agl, elevation)
    b = math.radians(float(yaw))
    return x + d * math.sin(b), y + d * math.cos(b)


def rectify_query(gray, K, agl, elevation, side_m, size=336):
    """Bird's-eye view of a gray image on a flat-ground square of side_m metres centred on the
    optical-axis ground point, camera heading up (the orientation of the rotated SCC map).

    Returns (image uint8 size x size, mask uint8 0/1); pixels not seen by the camera are 0 in both.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('OBLIQUE: expected a uint8 gray image')
    if not side_m > 0:
        raise ValueError('OBLIQUE: invalid block side')
    a = math.radians(-float(elevation))
    d = view_offset(agl, elevation)
    s = float(side_m) / size
    z = np.array([0.0, math.cos(a), -math.sin(a)])      # optical axis in (right, forward, up)
    x = np.array([1.0, 0.0, 0.0])                       # image right
    y = np.cross(z, x)                                   # image down
    world_to_cam = np.stack([x, y, z])
    ground = np.array([[s, 0.0, -s * (size / 2 - 0.5)],
                       [0.0, -s, d + s * (size / 2 - 0.5)],
                       [0.0, 0.0, -float(agl)]])         # output pixel (u, v, 1) -> ground point - camera
    M = np.asarray(K, np.float64) @ world_to_cam @ ground
    h, w = gray.shape
    out = cv2.warpPerspective(gray, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    p = M @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    with np.errstate(divide='ignore', invalid='ignore'):
        px, py = p[0] / p[2], p[1] / p[2]
    mask = ((p[2] > 0) & (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)).reshape(size, size).astype(np.uint8)
    out[mask == 0] = 0
    return out, mask


def apply_mask(tile, mask):
    """Zero the candidate-tile pixels the query cannot see (same relative footprint as the query)."""
    if mask is None:
        return tile
    tile = np.array(tile, copy=True)
    tile[mask == 0] = 0
    return tile
