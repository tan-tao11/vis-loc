import numpy as np
import pytest
from rasterio.transform import from_origin
from pyproj import Transformer
from common import config, ROOT
from geometry import GeoRaster, camera_rotation, reference_angle
from synthetic import write_tiff, make_fixture
from records import scan

@pytest.mark.parametrize('yaw', [0, 90, 180, 270])
def test_nadir_heading_sign_and_grid_convergence(tmp_path, yaw):
    f = tmp_path/'map.tif'; write_tiff(f, np.ones((100, 100), np.uint8), from_origin(500000, 4700000, 2, 2), 'EPSG:32645')
    c = config(ROOT/'configs/train.yaml')
    lon, lat = Transformer.from_crs(32645, 4326, always_xy=True).transform(500100, 4699900)
    r = dict(innsYaw=yaw, innsPitch=0, innsRoll=0, sensorYaw=0, sensorPitch=0, sensorRoll=0, isEO=0,
             droneLongitude=lon, droneLatitude=lat, droneAltitude=1000)
    R = camera_rotation(r, c['pose']); assert np.allclose(R[:, 2], [0, 0, 1])
    with GeoRaster(f) as sat:
        angle = reference_angle(r, R, sat, 0.)
        assert abs((angle-yaw+180) % 360-180) < .01

def test_per_image_adapter_and_mismatch(tmp_path):
    cf = make_fixture(tmp_path); c = config(cf); report = scan(c); assert not report['errors']
    for pair in report['pairs']:
        from pathlib import Path
        r = pair['record']; txt = '\n'.join(f'{k}={v}' for k, v in r.items() if not k.startswith('_'))
        Path(pair['image']).with_suffix('.txt').write_text(txt)
    c['data']['per_image_txt'] = True; c['data']['record_format'] = 'auto'
    assert len(scan(c)['pairs']) == 12 and not scan(c)['errors']
    p = Path(report['pairs'][0]['image']).with_suffix('.txt'); p.write_text(p.read_text().replace('number=1000.0', 'number=999.0'))
    assert any(e['reason'] == 'record_error' for e in scan(c)['errors'])
