import math


def estimate_reference_side(data, drone_size):
    """Flat-ground bounding square; no map reading or matching changes.

    Uses the existing effective optical elevation, K and query centre crop.
    roll_deg is the image roll under the calibrated EO camera/INS convention.
    For non-EO near-nadir images the caller supplies the small-roll approximation 0.
    """
    H = float(data['rel_height'])
    fx = float(data['focal_len'])
    fy = float(data['focal_y'])
    alpha = math.radians(-float(data['pitch']))
    roll = math.radians(float(data.get('roll_deg', 0)))
    scale = float(data.get('reference_window_scale', 1.2))
    if not all(math.isfinite(v) for v in (H, fx, fy, alpha, roll, scale)):
        raise ValueError('WINDOW: nonfinite input')
    if min(H, fx, fy, scale) <= 0 or not 0 < alpha <= math.pi/2 + 1e-6:
        raise ValueError('WINDOW: invalid AGL, intrinsics, elevation or scale')
    if data.get('is_slant', False) and not data.get('pose_known', False):
        raise ValueError('WINDOW: slant images require pose_priori=yp')
    # Match crop_center: odd requested sizes produce one fewer pixel.
    crop_w = 2 * (int(drone_size[0]) // 2)
    crop_h = 2 * (int(drone_size[1]) // 2)
    if min(crop_w, crop_h) <= 0:
        raise ValueError('WINDOW: empty query crop')
    tx, ty = crop_w/(2*fx), crop_h/(2*fy)
    cr, sr = abs(math.cos(roll)), abs(math.sin(roll))
    tx, ty = cr*tx + sr*ty, sr*tx + cr*ty
    if alpha <= math.atan(ty) + math.radians(2):
        raise ValueError('WINDOW: query approaches horizon; this approximation is unsuitable')
    sa, ca = math.sin(alpha), math.cos(alpha)
    near_m = H*(ca-sa*ty)/(sa+ca*ty)
    far_m = H*(ca+sa*ty)/(sa-ca*ty)
    width_m = 2*H*tx/(sa-ca*ty)
    height_m = far_m-near_m
    side_m = scale*max(width_m, height_m)
    print('[REF_WINDOW] AGL=%.1f alpha=%.2f roll=%.2f ground_W_H=(%.1f,%.1f)m scale=%.2f side=%.1fm'
          % (H, math.degrees(alpha), math.degrees(roll), width_m, height_m, scale, side_m), flush=True)
    return side_m
