"""NED active column rotations, independent raster georeferencing and ECEF rays."""
import math
from pathlib import Path
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.windows import Window
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

def checked_rotation(R):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(R), 1, atol=1e-7):
        raise ValueError('Rotation must be orthogonal with det=+1')
    return R

def euler_zyx(yaw, pitch, roll):
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    return checked_rotation([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                             [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])

def camera_rotation(record, pose):
    if pose['euler_order'] != 'ZYX': raise ValueError('Only explicit ZYX is implemented')
    ins = euler_zyx(*(record['inns'+a] for a in ('Yaw', 'Pitch', 'Roll')))
    if record['isEO'] == 1:
        return checked_rotation(ins @ checked_rotation(pose['eo_inns_from_cam']))
    if record['isEO'] != 0: raise ValueError('isEO must be 0 or 1')
    sensor = euler_zyx(*(record['sensor'+a] for a in ('Yaw', 'Pitch', 'Roll')))
    return checked_rotation(ins @ checked_rotation(pose['inns_from_mount']) @ sensor @ checked_rotation(pose['sensor_from_cam']))

def intrinsics(width, height, record, override=None):
    if override is not None:
        K = np.asarray(override, dtype=float)
    else:
        hf, vf = np.deg2rad([record['hfov'], record['vfov']])
        if not 0 < hf < np.pi or not 0 < vf < np.pi: raise ValueError('Invalid FOV')
        K = np.array([[width/(2*np.tan(hf/2)), 0, width/2], [0, height/(2*np.tan(vf/2)), height/2], [0, 0, 1.]])
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Invalid K')
    return K

def tfw_transform(filename):
    values = [float(s) for s in Path(filename).read_text().split()]
    if len(values) != 6: raise ValueError('TFW must contain A D B E C F')
    A, D, B, E, C, F = values
    return Affine(A, B, C-(A+B)/2, D, E, F-(D+E)/2)

class GeoRaster:
    def __init__(self, filename, authority='error'):
        self.filename = str(filename)
        # Read internal metadata only; otherwise GDAL may silently prefer sidecars.
        self.ds = rasterio.open(filename, GEOREF_SOURCES='INTERNAL')
        p = Path(filename); tfw = p.with_suffix('.tfw'); prj = p.with_suffix('.prj')
        embedded = self.ds.transform if self.ds.crs else None
        embedded_crs = CRS.from_user_input(self.ds.crs) if self.ds.crs else None
        external = tfw_transform(tfw) if tfw.exists() else None
        external_crs = CRS.from_wkt(prj.read_text(encoding='utf-8-sig')) if prj.exists() else None
        self.conflict = None
        transform_diff = embedded is not None and external is not None and not np.allclose(tuple(embedded)[:6], tuple(external)[:6], rtol=0, atol=1e-7)
        crs_diff = embedded_crs is not None and external_crs is not None and not embedded_crs.equals(external_crs)
        if transform_diff or crs_diff:
            self.conflict = {'embedded_affine': list(embedded)[:6] if embedded else None,
                             'sidecar_affine': list(external)[:6] if external else None,
                             'affine_delta': (np.array(embedded)[:6]-np.array(external)[:6]).tolist() if embedded and external else None,
                             'embedded_crs': str(embedded_crs), 'sidecar_crs': str(external_crs), 'crs_conflict': crs_diff}
            if authority == 'error':
                self.ds.close(); raise ValueError(f'Geo metadata conflict: {self.conflict}')
        if authority not in ('error', 'embedded', 'sidecar'): raise ValueError('Invalid geo authority')
        if authority == 'sidecar':
            self.transform, self.crs = external, external_crs
        elif authority == 'embedded':
            self.transform, self.crs = embedded, embedded_crs
        else:
            self.transform = embedded if embedded is not None else external
            self.crs = embedded_crs if embedded_crs is not None else external_crs
        if self.transform is None or self.crs is None:
            self.ds.close(); raise ValueError(f'Missing georeferencing: {filename}')
        if abs(self.transform.determinant) < 1e-15: raise ValueError('Singular raster transform')
        self.width, self.height = self.ds.width, self.ds.height
        self.from_wgs = Transformer.from_crs(4326, self.crs, always_xy=True)
        self.to_wgs = Transformer.from_crs(self.crs, 4326, always_xy=True)

    def close(self): self.ds.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def metadata(self):
        return {'path': self.filename, 'crs': self.crs.to_wkt(), 'affine_corner': list(self.transform)[:6],
                'width': self.width, 'height': self.height, 'dtype': self.ds.dtypes, 'count': self.ds.count,
                'nodata': self.ds.nodata, 'units': self.ds.units, 'tags': self.ds.tags(), 'conflict': self.conflict}

    def sample(self, lon, lat):
        x, y = self.from_wgs.transform(lon, lat)
        col, row = ~self.transform * (x, y)
        # Bilinear at pixel centres; all four neighbours must be valid.
        col -= .5; row -= .5
        i, j = math.floor(col), math.floor(row)
        if i < 0 or j < 0 or i+1 >= self.width or j+1 >= self.height: raise ValueError('DSM out_of_bounds')
        a = self.ds.read(1, window=Window(i, j, 2, 2), masked=True)
        if np.ma.getmaskarray(a).any() or not np.isfinite(a).all(): raise ValueError('DSM NoData')
        u, v = col-i, row-j
        return float(a[0, 0]*(1-u)*(1-v)+a[0, 1]*u*(1-v)+a[1, 0]*(1-u)*v+a[1, 1]*u*v)

class Ground:
    def __init__(self, geo):
        self.geo = geo
        self.raster = GeoRaster(geo['dsm'], geo['dsm_authority']) if geo.get('dsm') else None
        if not self.raster and (not geo['flat_fallback'] or geo['flat_ground_normal_m'] is None):
            raise ValueError('DSM required; no implicit zero-height fallback')
        if self.raster and geo['dsm_to_platform_offset_m'] is None: raise ValueError('DSM/platform datum conversion missing')

    def height(self, lon, lat):
        if self.raster:
            return self.raster.sample(lon, lat)*self.geo['dsm_unit_to_m']+self.geo['dsm_to_platform_offset_m']
        return float(self.geo['flat_ground_normal_m'])

    def close(self):
        if self.raster: self.raster.close()

class CameraFrame:
    def __init__(self, lon, lat, normal_height, normal_to_ellipsoid):
        if normal_to_ellipsoid is None: raise ValueError('Normal-to-ellipsoid conversion missing')
        self.offset = float(normal_to_ellipsoid)
        self.to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
        self.from_ecef = Transformer.from_crs(4978, 4979, always_xy=True)
        self.origin = np.array(self.to_ecef.transform(lon, lat, normal_height+self.offset))
        lo, la = np.deg2rad([lon, lat]); cl, sl, ca, sa = np.cos(lo), np.sin(lo), np.cos(la), np.sin(la)
        self.ecef_from_ned = checked_rotation([[-sa*cl, -sl, -ca*cl], [-sa*sl, cl, -ca*sl], [ca, 0, -sa]])

    def position(self, ned):
        lon, lat, alt = self.from_ecef.transform(*(self.origin+self.ecef_from_ned@np.asarray(ned)))
        return lon, lat, alt-self.offset

def intersect(frame, ray_ned, ground, step=10, max_distance=30000):
    ray = np.asarray(ray_ned, float); ray /= np.linalg.norm(ray)
    if ray[2] <= 1e-5: raise ValueError('Ray above/near horizon')
    if step <= 0 or max_distance <= step: raise ValueError('Invalid ray marching limits')
    def residual(t):
        lon, lat, alt = frame.position(t*ray)
        return alt-ground.height(lon, lat), (lon, lat, alt)
    f, _ = residual(0)
    if f <= 0: raise ValueError('Camera at/below ground')
    previous = 0.
    for t in np.arange(step, max_distance+step, step):
        f, pos = residual(t)
        if f <= 0:
            lo, hi = previous, t
            for _ in range(24):
                mid = (lo+hi)/2; fm, pos = residual(mid)
                if fm > 0: lo = mid
                else: hi = mid
            _, pos = residual((lo+hi)/2)
            return np.array(pos)
        previous = t
    raise ValueError('Ray did not intersect before max_distance')

def project_footprint(record, R, K, crop_box, geo, ground, satellite):
    frame = CameraFrame(record['droneLongitude_gt'], record['droneLatitude_gt'], record['droneAltitude_gt'], geo['normal_to_ellipsoid_m'])
    left, top, right, bottom = crop_box
    n = geo['boundary_samples_per_edge']
    corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    boundary = [a+(b-a)*t for a, b in zip(corners, np.roll(corners, -1, axis=0)) for t in np.linspace(0, 1, n, endpoint=False)]
    invK = np.linalg.inv(K)
    def hit(uv):
        llh = intersect(frame, R @ invK @ np.r_[uv, 1.], ground, geo['ray_step_m'], geo['max_ray_m'])
        return satellite.from_wgs.transform(llh[0], llh[1])
    points = [hit(p) for p in boundary]
    center = hit([(left+right)/2, (top+bottom)/2])
    poly = Polygon(points)
    if not poly.is_valid or poly.area <= 1: raise ValueError('Degenerate ground footprint')
    for x, y in points:
        col, row = ~satellite.transform * (x, y)
        if not 0 <= col < satellite.width or not 0 <= row < satellite.height: raise ValueError('Footprint outside satellite')
    return poly, center

def reference_angle(record, R, satellite, normal_to_ellipsoid):
    # Horizontal image-right projection remains stable at nadir. Uses prior, never GT.
    right = R[:, 0].copy(); right[2] = 0
    norm = np.linalg.norm(right)
    if norm < 1e-7: raise ValueError('Image-right heading singular; calibration required')
    frame = CameraFrame(record['droneLongitude'], record['droneLatitude'], record['droneAltitude'], normal_to_ellipsoid)
    ll0 = frame.position(np.zeros(3)); ll1 = frame.position(right/norm*10)
    p0 = np.array(~satellite.transform * satellite.from_wgs.transform(*ll0[:2]))
    p1 = np.array(~satellite.transform * satellite.from_wgs.transform(*ll1[:2]))
    dx, dy = p1-p0
    return float(np.rad2deg(np.arctan2(dy, dx)))
