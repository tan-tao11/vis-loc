import numpy as np
import pytest
from PIL import Image
from rasterio.transform import from_origin
from common import config, ROOT
from geometry import camera_rotation, checked_rotation, tfw_transform, GeoRaster, CameraFrame, Ground, intersect, reference_angle
from preprocess import intensity_u8
from synthetic import write_tiff

@pytest.fixture
def cfg(): return config(ROOT/'configs/train.yaml')

def test_axes_and_sensor(cfg):
    r = dict(innsYaw=0, innsPitch=0, innsRoll=0, sensorYaw=0, sensorPitch=0, sensorRoll=0, isEO=1)
    R = camera_rotation(r, cfg['pose']); assert np.allclose(R[:, 2], [1, 0, 0]); assert np.allclose(R[:, 0], [0, 1, 0]); assert np.allclose(R[:, 1], [0, 0, 1])
    r['isEO'] = 0; R = camera_rotation(r, cfg['pose']); assert np.allclose(R[:, 2], [0, 0, 1])
    assert np.allclose(R[:, 0], [0, 1, 0]); assert np.allclose(R[:, 1], [-1, 0, 0])
    r['sensorPitch'] = 10; assert not np.allclose(R, camera_rotation(r, cfg['pose']))
    with pytest.raises(ValueError): checked_rotation(np.diag([1, 1, -1]))

def test_ray_ground_horizon(cfg):
    g = Ground(dict(cfg['geo'], dsm=None, flat_fallback=True, flat_ground_normal_m=100))
    frame = CameraFrame(87, 42, 1000, 0)
    hit = intersect(frame, np.array([0., 0., 1.]), g, 20, 2000)
    assert np.allclose(hit, [87, 42, 100], atol=1e-4)
    for ray in ([1., 0., 0.], [1., 0., -0.1], [1., 0., 1e-7]):
        with pytest.raises(ValueError, match='horizon'): intersect(frame, ray, g)
    with pytest.raises(ValueError): Ground(dict(cfg['geo'], dsm=None))


def test_tfw_half_pixel_rotation_conflict(tmp_path):
    p = tmp_path/'x.tfw'; p.write_text('2\n.3\n.2\n-2\n101\n199\n')
    t = tfw_transform(p); assert np.allclose(t*(.5, .5), [101, 199]); assert t.b == .2 and t.d == .3
    write_tiff(tmp_path/'x.tif', np.ones((4, 4), np.uint8), from_origin(0, 10, 2, 2), 'EPSG:32645')
    with pytest.raises(ValueError, match='conflict'): GeoRaster(tmp_path/'x.tif')
    with GeoRaster(tmp_path/'x.tif', 'embedded') as r: assert r.conflict is not None

def test_dsm_bounds_nodata(tmp_path):
    a = np.ones((10, 10), np.float32)*50; a[4, 4] = -9999
    f = tmp_path/'dsm.tif'; write_tiff(f, a, from_origin(87, 42, .001, .001), 'EPSG:4326', -9999)
    with GeoRaster(f) as r:
        assert r.sample(87.002, 41.998) == 50
        with pytest.raises(ValueError, match='NoData'): r.sample(87.0045, 41.9955)
        with pytest.raises(ValueError, match='bounds'): r.sample(0, 0)

def test_uint16_not_truncated():
    a = np.array([[0, 32768, 65535]], np.uint16)
    with pytest.raises(ValueError): intensity_u8(a, {'uint16': None})
    out, _ = intensity_u8(a, {'uint16': [0, 65535]}); assert out.tolist() == [[0, 128, 255]]
