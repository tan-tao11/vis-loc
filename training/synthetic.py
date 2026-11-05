"""Small synthetic GIS fixture. Never claims real-world calibration or accuracy."""
import argparse
import copy
from pathlib import Path
import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from pyproj import Transformer
from shapely.geometry import box, mapping
import yaml
from common import config, write_json

def write_tiff(filename, a, transform, crs, nodata=None):
    with rasterio.open(filename, 'w', driver='GTiff', width=a.shape[1], height=a.shape[0], count=1,
                       dtype=a.dtype, transform=transform, crs=crs, nodata=nodata) as dst:
        dst.write(a, 1)

def make_fixture(directory):
    root = Path(directory).resolve(); root.mkdir(parents=True, exist_ok=True)
    c = copy.deepcopy(config(Path(__file__).parent/'configs/train.yaml'))
    c.update(synthetic=True, output=str(root/'run'))
    c['data'].update(root=str(root/'data'), record_format='csv', mapping_confirmed=True)
    c['pose'].update(calibration_confirmed=False, version='SYNTHETIC_candidate_axes_v1')
    c['preprocess']['deployment_confirmed'] = False
    c['geo'].update(satellite=str(root/'satellite.tif'), dsm=str(root/'dsm.tif'), vertical_confirmed=False,
                    normal_to_ellipsoid_m=0., dsm_to_platform_offset_m=0., platform_datum='synthetic normal', dsm_datum='synthetic normal')
    c['candidates'].update(search_radius_m=180, sizes_m=[160, 220], height_offsets_m=[0], max_per_query=100)
    c['split'].update(regions_geojson=str(root/'regions.geojson'), buffer_m=30)
    c['train'].update(epochs=8, patience=4, batch_size=4)
    c['extract'].update(batch_size=4)
    rng = np.random.default_rng(c['seed'])
    a = rng.integers(15, 240, (512, 1536), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (0, 0), 3)
    for _ in range(200):
        p = tuple(rng.integers([0, 0], [1536, 512])); q = tuple(rng.integers([0, 0], [1536, 512]))
        cv2.line(a, p, q, int(rng.integers(25, 230)), int(rng.integers(1, 7)))
    transform = from_origin(500000, 4701024, 2, 2)
    write_tiff(root/'satellite.tif', a, transform, 'EPSG:32645')
    # Consistent sidecars with pixel-centre C/F.
    (root/'satellite.tfw').write_text('2\n0\n0\n-2\n500001\n4701023\n')
    (root/'satellite.prj').write_text(rasterio.crs.CRS.from_epsg(32645).to_wkt())
    bounds = transform_bounds('EPSG:32645', 'EPSG:4326', 499500, 4699500, 503600, 4701500)
    dsm_tf = from_origin(bounds[0], bounds[3], (bounds[2]-bounds[0])/250, (bounds[3]-bounds[1])/150)
    write_tiff(root/'dsm.tif', np.full((150, 250), 100, np.float32), dsm_tf, 'EPSG:4326', -9999)
    regions = []
    to_ll = Transformer.from_crs(32645, 4326, always_xy=True)
    for i, split in enumerate(('train', 'val', 'test')):
        regions.append({'type': 'Feature', 'properties': {'split': split}, 'geometry': mapping(box(500000+1024*i, 4700000, 501024+1024*i, 4701024))})
    write_json(root/'regions.geojson', {'type': 'FeatureCollection', 'features': regions})
    import csv
    for gi, (group, spec) in enumerate(c['data']['groups'].items()):
        records = []
        for si, split in enumerate(('train', 'val', 'test')):
            seq = 'sequence_'+split; folder = root/'data'/spec['directory']/seq; folder.mkdir(parents=True, exist_ok=True)
            x, y = 500512+si*1024, 4700512
            lon, lat = to_ll.transform(x, y)
            r = dict(sequence=seq, number=1000, droneLongitude=lon, droneLatitude=lat, droneAltitude=220,
                     droneLongitude_gt=lon, droneLatitude_gt=lat, droneAltitude_gt=220, droneYaw=0, dronePitch=0,
                     droneRoll=0, innsYaw=gi*35, innsPitch=-75 if spec['expected_isEO'] else 0,
                     innsRoll=0, sensorYaw=0 if group != 'NV' else .1977569,
                     sensorPitch=0 if group != 'NV' else -.6976531, sensorRoll=0 if group != 'NV' else .7965331,
                     isEO=spec['expected_isEO'], hfov=55, vfov=45, ccdsize=12)
            # Diagnostic source imagery, not physically rendered IR. Keep explicitly synthetic.
            tile = a[220:292, 220+si*512:292+si*512]
            tile = cv2.resize(tile, (160, 128)); tile = np.clip(tile.astype(float)*(0.75 if gi > 1 else 1.1), 0, 255).astype(np.uint8)
            cv2.imencode('.png', tile)[1].tofile(str(folder/'1000.png'))
            records.append(r)
        with open(root/'data'/spec['records'], 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    cfg_path = root/'synthetic.yaml'; cfg_path.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return cfg_path

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default='artifacts/synthetic'); a = p.parse_args(); print(make_fixture(a.output))
