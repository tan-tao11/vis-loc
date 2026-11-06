"""Slant pipeline end to end on physically rendered oblique images (flat ground, synthetic map).

Checks the chain prepare -> TileSource: view-centred search window, bird's-eye query, masked tiles,
and that the best-matching candidate tile is a labelled positive.
"""
import copy
import csv
import math
from pathlib import Path
import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from common import config, ROOT
from synthetic import make_fixture
from prepare_dataset import prepare
from scc_inference import TileSource, camera_matrix, map_to_rotated
import scc_oblique

YAW, PITCH, ALT, GROUND = 90., -40., 220., 100.


def render(sat, transform, x, y, w=160, h=128, hfov=55., vfov=45.):
    K = camera_matrix({}, {'hfov': hfov, 'vfov': vfov}, w, h)
    a, b = math.radians(-PITCH), math.radians(YAW)
    f = np.array([math.sin(b), math.cos(b), 0.]); r = np.array([math.cos(b), -math.sin(b), 0.])
    z = math.cos(a)*f - math.sin(a)*np.array([0, 0, 1.]); R = np.stack([r, np.cross(z, r), z], 1)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d = R @ np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    s = (ALT-GROUND) / -d[2]
    inv = ~transform
    col, row = inv * (x + s*d[0], y + s*d[1])
    img = cv2.remap(sat, (np.asarray(col)-.5).astype(np.float32).reshape(h, w), (np.asarray(row)-.5).astype(np.float32).reshape(h, w),
                    cv2.INTER_LINEAR, borderValue=0)
    return img


def slant_fixture(tmp_path):
    c = config(make_fixture(tmp_path/'fx'))
    root = Path(c['data']['root'])
    with rasterio.open(c['geo']['satellite']) as ds: sat, tf = ds.read(1), ds.transform
    to_utm = Transformer.from_crs(4326, 32645, always_xy=True)
    for g in ('DS', 'NS'):
        spec = c['data']['groups'][g]
        rows = list(csv.DictReader(open(root/spec['records'], encoding='utf-8')))
        for r in rows:
            r.update(innsYaw=YAW, innsPitch=PITCH, innsRoll=0, droneAltitude=ALT, droneAltitude_gt=ALT)
            x, y = to_utm.transform(float(r['droneLongitude_gt']), float(r['droneLatitude_gt']))
            img = render(sat, tf, x - 150., y)      # look east from 150 m west of the region centre
            lon, lat = to_utm.transform(x - 150., y, direction='INVERSE')
            r.update(droneLongitude=lon, droneLatitude=lat, droneLongitude_gt=lon, droneLatitude_gt=lat)
            cv2.imencode('.png', img)[1].tofile(str(root/spec['directory']/r['sequence']/f"{r['number']}.png"))
        with open(root/spec['records'], 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    c['data']['groups'] = {g: c['data']['groups'][g] for g in ('DS', 'NS')}
    c['inference']['camera'] = {'reference_window_scale': 1.2, 'nadir_reference_alignment': False, 'oblique': True}
    c['inference']['camera_by_group'] = {}
    c['preprocess']['query_crop_scale'] = 0.5
    c['labels']['max_area_ratio'] = 8.0
    c['candidates'].update(train_window_m=500, eval_window_m=500, center_offset_m=20, altitude_error_m=0)
    c['output'] = str(tmp_path/'run')
    return c


def test_oblique_window_query_and_tiles(tmp_path):
    c = slant_fixture(tmp_path)
    m = prepare(copy.deepcopy(c))
    assert not m['failures'], m['failures'][:2]
    tiles = TileSource(c)
    try:
        for g in m['groups']:
            f = tiles.frame(g)
            # window centred ahead: view centre within one pixel-grid cell of the map centre
            d = scc_oblique.view_offset(f['agl'], f['elevation'])
            assert 100 < d < 160 and abs(f['agl'] - (ALT-GROUND)) < 1
            q = tiles.query(g, 336)
            mask = f['_mask']; assert 0.1 < mask.mean() < 1
            labels = np.array([x['label'] for x in g['candidates']])
            assert (labels == 1).any()
            for k in range(len(labels)):
                assert (tiles.tile(g, k, 336)[mask == 0] == 0).all()
            # the bird's-eye query matches the rotated map exactly at the optical-axis ground point
            r = g['record']; x0, y0 = Transformer.from_crs(4326, 32645, always_xy=True).transform(r['droneLongitude_gt'], r['droneLatitude_gt'])
            vx, vy = scc_oblique.view_centre(x0, y0, f['agl'], f['yaw'], f['elevation'])
            col, row = map_to_rotated(f, [[vx, vy]])[0]
            side = f['side_px']; r0, c0 = int(round(row - side/2)), int(round(col - side/2))
            ref = cv2.resize(f['ref_image'][r0:r0+side, c0:c0+side], (336, 336))
            ref = ref[..., 0] if ref.ndim == 3 else ref
            corr = np.corrcoef(q[mask > 0].astype(float), ref[mask > 0].astype(float))[0, 1]
            assert corr > 0.8, (g['id'], corr)
            # the grid block nearest the view centre is a positive
            near = int(np.argmin([math.hypot(*(np.array(x['center_map']) - [vx, vy])) for x in g['candidates']]))
            assert labels[near] == 1, (g['id'], near)
    finally:
        tiles.close()


def test_oblique_extract_train_evaluate(tmp_path, monkeypatch):
    """prepare -> extract (rectified query, masked tiles) -> train -> evaluate on the slant fixture."""
    import extract_features as extraction
    import cache as cache_module
    from tests.test_end_to_end import fake_backbone
    from train import train
    from evaluate import evaluate_arrays, load_heads_checkpoint
    fake_backbone(monkeypatch)
    c = slant_fixture(tmp_path); c['train'].update(epochs=3, patience=3, device='cpu')
    c['split'] = dict(c['split'], mode='sequence', regions_geojson=None,
                      sequence_splits={'sequence_train': 'train', 'sequence_val': 'val', 'sequence_test': 'test'})
    m = prepare(copy.deepcopy(c)); manifest = tmp_path/'run/manifest.json'; features = tmp_path/'run/features'
    extraction.extract(c, manifest, features, 'cpu'); extraction.merge(c, manifest, features, 1)
    best = train(c, manifest, features, tmp_path/'run/training', 'cpu')
    heads, _ = load_heads_checkpoint(best)
    _, arrays, _ = cache_module.load_cache(manifest, features)
    report = evaluate_arrays(m, arrays, heads, 'val', 'cpu')
    assert {'gem1024', 'heads'} <= set(report['results'])
    # the cached query feature is that of the rectified query, not of the centre crop
    from preprocess import query_image, tensor_image
    tiles = TileSource(c); g = m['groups'][0]
    try:
        rect, crop = tiles.query(g, 336), query_image(g['image'], c['preprocess'])[0]
    finally:
        tiles.close()
    f = lambda u: extraction.forward_frozen(None, tensor_image(u, c['preprocess']).unsqueeze(0), True)[0].numpy()
    assert np.allclose(arrays[g['id']][0], f(rect), atol=1e-5) and not np.allclose(arrays[g['id']][0], f(crop), atol=1e-3)
