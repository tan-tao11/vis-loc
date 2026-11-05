"""Deployed SCC infrared retrieval preprocessing, copied VERBATIM from inference/.

scc_geodata.py is a byte copy of inference/single-so/python/geodata.py. The six
functions below are verbatim copies (tests/test_inference_alignment.py compares
them with inference/ whenever that directory is present). Do not edit them here:
change inference/ first, then re-copy. Training-only helpers come after the copies.
"""
import contextlib
import io
import math
import cv2
import numpy as np
from PIL import Image

VERBATIM = {'camera_matrix': 'single-so/python/backends.py', 'optical_pose': 'single-so/python/backends.py',
            'dumpRotateImage': 'projects/SCC-Loc/utils.py', 'estimate_reference_side': 'projects/SCC-Loc/utils.py',
            'compute_block_mid_wo_black': 'projects/SCC-Loc/utils.py', 'crop_center': 'projects/SCC-Loc/utils.py'}


# ---- verbatim from backends.py ----
def camera_matrix(camera, t, width, height):
    if "K" in camera:
        K = np.asarray(camera["K"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("invalid camera K")
        if not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("invalid camera K last row")
        calibrated_w, calibrated_h = map(int, camera["image_size"])
        if min(calibrated_w, calibrated_h) <= 0:
            raise ValueError("invalid calibration image_size [width,height]")
        K = np.diag([width/calibrated_w, height/calibrated_h, 1]) @ K
    else:
        hfov, vfov = t["hfov"], t["vfov"]
        if not (0 < hfov < 179 and 0 < vfov < 179):
            raise ValueError("valid hfov/vfov or calibrated camera.K is required")
        K = np.array([[width/(2*math.tan(math.radians(hfov)/2)), 0, (width-1)/2],
                      [0, height/(2*math.tan(math.radians(vfov)/2)), (height-1)/2], [0, 0, 1]], dtype=float)
    return K


# ---- verbatim from backends.py ----
def optical_pose(t, is_eo, camera):
    """Body-to-NED Rz(yaw)Ry(pitch)Rx(roll), with an explicit optical axis in INS axes."""
    # Conversation-reconstructed near-nadir reference alignment, not full extrinsics.
    # The 1-degree gates are approximation checks, not measured calibration accuracy.
    if camera.get("nadir_reference_alignment", False):
        if is_eo:
            if abs(t["inns_pitch"] + 90.0) > 1.0:
                raise ValueError("EO nadir alignment requires innsPitch near -90; disable it for slant")
            return (t["inns_yaw"] + t["inns_roll"]) % 360.0, -90.0
        if any(abs(t[k]) > 1.0 for k in (
                "sensor_yaw", "sensor_pitch", "sensor_roll", "inns_pitch", "inns_roll")):
            raise ValueError("non-EO nadir alignment requires near-zero sensor angles and INS pitch/roll")
        return t["inns_yaw"] % 360.0, -90.0
    if "optical_axis_in_inns" in camera:
        axis = np.asarray(camera["optical_axis_in_inns"], dtype=float)
    elif is_eo:
        axis = np.array([1., 0., 0.])  # Interface explicitly states EO optical axis aligns with INS X.
    else:
        raise ValueError("yp prior for non-EO needs calibrated optical_axis_in_inns")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("invalid optical_axis_in_inns")
    yaw, pitch, roll = np.deg2rad([t["inns_yaw"], t["inns_pitch"], t["inns_roll"]])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                  [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
    n, e, d = R @ (axis / np.linalg.norm(axis))
    elevation = math.degrees(math.atan2(-d, math.hypot(n, e)))
    if elevation >= -1:
        raise ValueError("optical axis must point below the horizon for map localization")
    return math.degrees(math.atan2(e, n)) % 360, elevation


# ---- verbatim from SCC-Loc/utils.py ----
def dumpRotateImage(img, degree):
    """rotate the image by degree"""
    height, width = img.shape[:2]
    radians = np.radians(degree)
    heightNew = int(abs(height * np.cos(radians)) + abs(width * np.sin(radians)))
    widthNew = int(abs(width * np.cos(radians)) + abs(height * np.sin(radians)))
    
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if len(img.shape) == 3:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        img_pil = Image.fromarray(img)

    img_rotated_pil = img_pil.rotate(degree, expand=True, resample=Image.BICUBIC)
    imgRotation = np.asarray(img_rotated_pil)
    
    if imgRotation.shape[0] != heightNew or imgRotation.shape[1] != widthNew:
       heightNew, widthNew = imgRotation.shape[:2]

    if len(imgRotation.shape) == 3:
        imgRotation = cv2.cvtColor(imgRotation, cv2.COLOR_RGB2BGR)

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1.0)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    return imgRotation, matRotation


# ---- verbatim from SCC-Loc/utils.py ----
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


# ---- verbatim from SCC-Loc/utils.py ----
def compute_block_mid_wo_black(image, block_size, step_size):
    """compute retrieval block mid points while avoiding black borders in the reference image"""
    # use thumbnails to increase computing speed
    scale = 5
    thresold = 0.9 # valid pixel threshold to filter out black borders
    h, w = image.shape[:2]
    
    target_h = max(1, int(h / scale))
    target_w = max(1, int(w / scale))
    small_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def get_clean_starts(length, blk_len, step):
        starts = list(range(0, length - blk_len + 1, step))
        if len(starts) == 0 or starts[-1] < length - blk_len:
            starts.append(length - blk_len)
        return sorted(list(set(starts)))

    x_starts = get_clean_starts(h, block_size[0], step_size[0])
    y_starts = get_clean_starts(w, block_size[1], step_size[1])

    mids = []
    
    for start_x in x_starts:
        for start_y in y_starts:
            end_x = start_x + block_size[0]
            end_y = start_y + block_size[1]
            
            s_sx, s_ex = int(start_x / scale), int(end_x / scale)
            s_sy, s_ey = int(start_y / scale), int(end_y / scale)
            
            if s_ex <= s_sx: s_ex = s_sx + 1
            if s_ey <= s_sy: s_ey = s_sy + 1
            
            block_small = small_img[s_sx:s_ex, s_sy:s_ey]
            if block_small.size == 0: continue
            
            if len(block_small.shape) == 3:
                valid_mask = np.any(block_small > 0, axis=-1)
            else:
                valid_mask = block_small > 0
                
            valid_ratio = np.sum(valid_mask) / valid_mask.size
            
            if valid_ratio >= thresold:
                mid_x = (start_x + end_x) / 2.0
                mid_y = (start_y + end_y) / 2.0
                mids.append([mid_x, mid_y])

    if len(mids) == 0:
        return np.array([[h/2.0, w/2.0]])
        
    return np.array(mids)


# ---- verbatim from SCC-Loc/utils.py ----
def crop_center(image_path, width, height):
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]
    center_x, center_y = original_width // 2, original_height // 2
    left = int(center_x - width // 2)
    top = int(center_y - height // 2)
    right = int(center_x + width // 2)
    bottom = int(center_y + height // 2)

    left = max(0, left)
    top = max(0, top)
    right = min(original_width, right)
    bottom = min(original_height, bottom)

    cropped_img = img[top:bottom, left:right]
    return Image.fromarray(cropped_img)


# =====================================================================================
# Training-only helpers. They only call the verbatim functions above in the same order
# as SCCBackend.prepare/localize -> utils.retrieval_all -> feature_extract.extract_features.
# =====================================================================================

def telemetry(record):
    """Training record -> inference telemetry dict. Uses the PRIOR position, never *_gt."""
    return dict(lon=float(record['droneLongitude']), lat=float(record['droneLatitude']),
                altitude=float(record['droneAltitude']), drone_yaw=float(record.get('droneYaw', 0.)),
                drone_pitch=float(record.get('dronePitch', 0.)), drone_roll=float(record.get('droneRoll', 0.)),
                sensor_yaw=float(record['sensorYaw']), sensor_pitch=float(record['sensorPitch']),
                sensor_roll=float(record['sensorRoll']), inns_yaw=float(record['innsYaw']),
                inns_pitch=float(record['innsPitch']), inns_roll=float(record['innsRoll']),
                hfov=float(record['hfov']), vfov=float(record['vfov']), ccd_size=float(record.get('ccdsize', 0.)))


def reference_config(c, window_m):
    """naviParams infrared.reference equivalent; image/dsm come from geo.satellite/geo.dsm."""
    from common import path
    ref = dict(c['inference']['reference'])
    ref.update(image=str(path(c['geo']['satellite'])), dsm=str(path(c['geo']['dsm'])), window_m=float(window_m))
    ref.setdefault('georeference', 'auto')
    if ref['georeference'] != 'auto': raise ValueError('Training mirrors georeference: auto (GeoReference) only')
    return ref


def open_reference(c, window_m):
    """GeoReference enters a rasterio.Env in __init__ and leaves it in close(); with several
    references open the LIFO env stack breaks. Leave it right away (it only sets GDAL cache size
    and PROJ_NETWORK=OFF, which never change pixel values) so references close in any order."""
    import os
    from scc_geodata import GeoReference
    os.environ.setdefault('PROJ_NETWORK', 'OFF')
    r = GeoReference(reference_config(c, window_m))
    if r.env is not None: r.env.__exit__(None, None, None); r.env = None
    return r


def camera_for(c, group):
    camera = dict(c['inference'].get('camera', {}))
    camera.update(c['inference'].get('camera_by_group', {}).get(group, {}))
    return camera


def frame_retrieval(full_ref, t, width, height, is_eo, is_slant, camera, pose_priori='yp', cover=50, window_m=None):
    """Search map, rotation and candidate block centres exactly as the deployed SCC backend.

    t holds the search-centre prior (already passed through prior tracking, as in BaseBackend.prepare);
    window_m is the per-frame search window the tracker chose (inference sets full_ref.window_m likewise).
    """
    if pose_priori not in ('yp', 'unknown'): raise ValueError('pose_priori must be yp or unknown')
    if window_m is not None: full_ref.window_m = float(window_m)
    view = full_ref.for_frame(t)
    K = camera_matrix(camera, t, width, height)
    agl = view.agl(t, camera)
    yaw, elevation = (0., -90.) if pose_priori == 'unknown' else optical_pose(t, bool(is_eo), camera)
    data = dict(rel_height=agl, yaw=yaw, pitch=elevation, width=width, height=height,
                cam_size=math.hypot(width, height), focal_len=float(K[0, 0]), focal_y=float(K[1, 1]),
                roll_deg=float(t['inns_roll']) if bool(is_eo) else 0.0,
                reference_window_scale=float(camera.get('reference_window_scale', 1.2)),
                is_slant=bool(is_slant), pose_known=(pose_priori == 'yp'), map_hw=tuple(view.image.shape[:2]))
    if pose_priori == 'yp': ref_image, rotation = dumpRotateImage(view.image, yaw)
    else: ref_image, rotation = view.image, np.eye(2, 3, dtype=np.float32)
    side = min(width, height)
    with contextlib.redirect_stdout(io.StringIO()):  # silence per-frame [REF_WINDOW] prints
        side_m = estimate_reference_side(data, [side, side])
    a = math.radians(yaw if pose_priori == 'yp' else 0.0)
    need = max(1.1*side_m*(abs(math.cos(a)) + abs(math.sin(a))), 2.0*side_m)   # = BaseBackend.window_for_block
    if full_ref.window_m > 0 and need > full_ref.window_m:
        return frame_retrieval(full_ref, t, width, height, is_eo, is_slant, camera, pose_priori, cover, window_m=need)
    side_px = max(10, 2 * int(math.ceil(side_m / (2 * view.res))))
    angle = math.radians(yaw if pose_priori == 'yp' else 0.0)
    required_px = int(math.ceil(side_px * (abs(math.cos(angle)) + abs(math.sin(angle)))))
    if side_px > min(ref_image.shape[:2]) or required_px > min(data['map_hw']):
        raise ValueError('REF_WINDOW: block %.1fm larger than search map; increase window_m' % side_m)
    step = max(1, int(side_px * (100 - cover) / 100))
    mids = compute_block_mid_wo_black(ref_image, [side_px, side_px], [step, step])
    return dict(view=view, ref_image=ref_image, rotation=rotation, side_px=side_px, side_m=float(side_m), window_m=float(full_ref.window_m),
                mids=np.asarray(mids, dtype=float).reshape(-1, 2), K=K, agl=float(agl), yaw=float(yaw),
                elevation=float(elevation))


def block_bounds(mid, side_px):
    """Same integer slicing as feature_extract.extract_features: rows r0:r1, cols c0:c1."""
    mx, my = mid
    return int(mx - side_px / 2), int(mx + side_px / 2), int(my - side_px / 2), int(my + side_px / 2)


def crop_tile(ref_image, mid, side_px, size=336):
    r0, r1, c0, c1 = block_bounds(mid, side_px)
    return cv2.resize(ref_image[r0:r1, c0:c1], (size, size))


def tile_tensor(tile, mean, std):
    """Image.fromarray + ToTensor + Normalize (channel order kept, as in inference)."""
    import torch
    a = np.asarray(tile, dtype=np.float32) / 255.
    if a.ndim == 2: a = np.repeat(a[..., None], 3, axis=-1)
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def block_map_corners(frame, mid):
    """Block corners in working-CRS metres, via the inverse of the inference rotation matrix."""
    r0, r1, c0, c1 = block_bounds(mid, frame['side_px'])
    inverse = cv2.invertAffineTransform(np.asarray(frame['rotation'], np.float64))
    pts = np.array([[c0, r0, 1], [c1, r0, 1], [c1, r1, 1], [c0, r1, 1]], float) @ inverse.T
    view = frame['view']
    return np.stack([view.x0 + pts[:, 0] * view.res, view.y0 - pts[:, 1] * view.res], -1)


def map_to_rotated(frame, xy):
    """Working-CRS metres -> pixel (col,row) in the rotated search map."""
    view = frame['view']; xy = np.asarray(xy, float).reshape(-1, 2)
    uv = np.stack([(xy[:, 0] - view.x0) / view.res, (view.y0 - xy[:, 1]) / view.res, np.ones(len(xy))], -1)
    return uv @ np.asarray(frame['rotation'], np.float64).T


class TileSource:
    """Rebuilds candidate tiles from a manifest group; caches the last group's rotated map."""
    def __init__(self, c):
        self.c = c; self.ref = None; self._key = None; self._frame = None

    def frame(self, g):
        if self._key != g['id']:
            if self.ref is None: self.ref = open_reference(self.c, g['window_m'])
            inf = self.c['inference']
            self._frame = frame_retrieval(self.ref, g['telemetry'], g['width'], g['height'], g['record']['isEO'],
                                          g['is_slant'], camera_for(self.c, g['group']), inf.get('pose_priori', 'yp'),
                                          inf.get('retrieval_overlap_percent', 50), window_m=g['window_m'])
            if self._frame['side_px'] != g['side_px'] or len(self._frame['mids']) != g['grid_size']:
                raise ValueError('Rebuilt search grid differs from manifest; regenerate manifest: ' + g['id'])
            self._key = g['id']
        return self._frame

    def tile(self, g, index, size=336):
        f = self.frame(g)
        return crop_tile(f['ref_image'], g['candidates'][index]['mid'], f['side_px'], size)

    def close(self):
        if self.ref is not None: self.ref.close()
        self.ref = None
