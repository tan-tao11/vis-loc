"""Retrieval / PnP consistency check and retrieval-based fallback position (SCC infrared).

A PnP fix is trusted only if the ground point it implies falls inside the retrieved block it was
matched against and its altitude agrees with the prior. When no fix passes, the retrieval itself can
give a position (vertical camera: block centre ~= aircraft; accuracy ~ half the block grid step).

All coordinates are working-CRS metres; yaw is the camera heading (clockwise from grid north),
elevation the optical-axis elevation in degrees (-90 = nadir).
"""
import math
import numpy as np
from oblique import view_offset

DEFAULTS = dict(enabled=False, block_ratio=0.75, altitude_m=300., retrieval_fallback=False,
                top_k=3, ambiguous_gap=0.005, max_fallback_agl=2000.)


def settings(cfg):
    c = dict(DEFAULTS, **(cfg or {}))
    if c['block_ratio'] <= 0 or c['altitude_m'] <= 0 or int(c['top_k']) < 1:
        raise ValueError('consistency: block_ratio, altitude_m and top_k must be positive')
    return c


def block_centre(rotation, row, col, height, width, x0, y0, res):
    """Working-CRS centre of a retrieval block given in the rotated search map."""
    inv = np.asarray(__import__('cv2').invertAffineTransform(np.asarray(rotation, np.float64)))
    ux, uy = inv @ np.array([col + width / 2 - .5, row + height / 2 - .5, 1.0])
    return float(x0 + ux * res), float(y0 - uy * res)


def _ahead(agl, elevation):
    return 0.0 if elevation <= -89.9 else view_offset(max(float(agl), 1.0), elevation)


def check_fix(x, y, height, prior_altitude, centre, side_m, agl, yaw, elevation, cfg):
    """(ok, reason, distance_m): is the PnP fix (x, y, height) consistent with its retrieval block?"""
    dz = float(height) - float(prior_altitude)
    if abs(dz) > cfg['altitude_m']:
        return False, 'altitude', float('nan')
    d = _ahead(agl + dz, elevation)                    # look-ahead with the fix's own height
    b = math.radians(yaw)
    vx, vy = x + d * math.sin(b), y + d * math.cos(b)  # ground point the fix says the camera sees
    dist = math.hypot(vx - centre[0], vy - centre[1])
    if dist > cfg['block_ratio'] * side_m:
        return False, 'outside_block', dist
    return True, 'ok', dist


def retrieval_position(centres, scores, side_m, agl, yaw, elevation, cfg):
    """Aircraft (x, y) from the top retrieval blocks, or None if the ranking is ambiguous."""
    k = min(int(cfg['top_k']), len(centres))
    if k == 0:
        return None
    c = np.asarray(centres[:k], float); s = np.asarray(scores[:k], float)
    near = np.hypot(*(c - c[0]).T) <= 1.01 * side_m
    if np.any(~near & (s >= s[0] - cfg['ambiguous_gap'])):
        return None                                     # a distant block scores as well as the best
    w = np.exp((s[near] - s[0]) / 0.01)
    gx, gy = (c[near] * w[:, None]).sum(0) / w.sum()
    d = _ahead(agl, elevation); b = math.radians(yaw)
    return float(gx - d * math.sin(b)), float(gy - d * math.cos(b))
