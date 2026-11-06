"""Bird's-eye rectification against a physically rendered oblique view of a known map (flat ground)."""
import math
import cv2
import numpy as np
import pytest
from scc_inference import dumpRotateImage, camera_matrix
from scc_oblique import rectify_query, view_offset

RES, N = 2.0, 4000


def scene():
    rng = np.random.default_rng(0)
    sat = cv2.GaussianBlur(rng.integers(0, 255, (N, N)).astype(np.uint8), (0, 0), 2)
    for _ in range(300):
        p = tuple(map(int, rng.integers(0, N, 2))); q = tuple(map(int, rng.integers(0, N, 2)))
        cv2.line(sat, p, q, int(rng.integers(0, 255)), int(rng.integers(2, 9)))
    return sat


def render(sat, yaw, elev, H, w=640, h=512, hfov=28.5, vfov=22.8):
    K = camera_matrix({}, {'hfov': hfov, 'vfov': vfov}, w, h)
    a, b = math.radians(-elev), math.radians(yaw)
    f = np.array([math.sin(b), math.cos(b), 0.]); r = np.array([math.cos(b), -math.sin(b), 0.])
    z = math.cos(a)*f - math.sin(a)*np.array([0, 0, 1.]); R = np.stack([r, np.cross(z, r), z], 1)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d = R @ np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    s = H / -d[2]
    col = (s*d[0]/RES + N/2 - .5).astype(np.float32).reshape(h, w); row = (N/2 - s*d[1]/RES - .5).astype(np.float32).reshape(h, w)
    return cv2.remap(sat, col, row, cv2.INTER_LINEAR, borderValue=0), K


@pytest.mark.parametrize('yaw,elev,H', [(27.7, -15.35, 900.), (333.1, -17.3, 1100.), (120., -25., 800.)])
def test_rectified_query_matches_rotated_map(yaw, elev, H):
    sat = scene(); img, K = render(sat, yaw, elev, H); side_m = 2500.
    q, m = rectify_query(img, K, H, elev, side_m)
    rot, M = dumpRotateImage(sat, yaw)
    D = view_offset(H, elev); E, Nn = D*math.sin(math.radians(yaw)), D*math.cos(math.radians(yaw))
    c = M @ np.array([E/RES + N/2 - .5, N/2 - Nn/RES - .5, 1.])
    sp = int(round(side_m/RES)); r0, c0 = int(round(c[1] - sp/2)), int(round(c[0] - sp/2))
    corr = lambda dr: np.corrcoef(q[m > 0].astype(float), cv2.resize(rot[r0+dr:r0+dr+sp, c0:c0+sp], (336, 336))[m > 0].astype(float))[0, 1]
    assert m.mean() > 0.2
    assert corr(0) > 0.9 and corr(30) < corr(0) - 0.2
