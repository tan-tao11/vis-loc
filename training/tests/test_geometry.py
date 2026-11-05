import numpy as np
import pytest
from PIL import Image
from rasterio.transform import from_origin
from common import config, ROOT
from geometry import camera_rotation, checked_rotation, tfw_transform, GeoRaster, CameraFrame, Ground, intersect, reference_angle
from preprocess import rotation_affine, local_rotated_crop, intensity_u8
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

@pytest.mark.parametrize('angle', [0, 90, 180, 270, 27.6, -36.4])
def test_pil_local_rotation_exact(tmp_path, angle):
    # Asymmetric directional image, not a rotationally symmetric 180-only fixture.
    a = np.zeros((63, 91), np.uint8); a[8:18, 9:50] = 220; a[18:40, 9:17] = 100; a[41:48, 70:79] = 170
    f = tmp_path/'a.tif'; write_tiff(f, a, from_origin(500000, 4700000, 2, 2), 'EPSG:32645')
    expected = np.asarray(Image.fromarray(a).rotate(angle, expand=True, resample=Image.Resampling.BICUBIC))
    M, size = rotation_affine(91, 63, angle); assert size == expected.shape[::-1]
    with GeoRaster(f) as r:
        box = [3, 2, size[0]-4, size[1]-3]
        actual, _, _ = local_rotated_crop(r, M, box, {'uint8': [0, 255]})
    assert np.max(np.abs(actual.astype(int)-expected[2:size[1]-3, 3:size[0]-4].astype(int))) <= 1
    p = np.array([13.5, 25.5, 1]); assert np.allclose(np.linalg.inv(M)@M@p, p)
    if angle == 90: assert np.array_equal(expected, np.rot90(a))
    if angle == 270: assert np.array_equal(expected, np.rot90(a, 3))

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
