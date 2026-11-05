"""Training must use the deployed SCC preprocessing verbatim (skipped where inference/ is absent)."""
import ast
from pathlib import Path
import numpy as np
import pytest
import scc_inference
from common import ROOT

INFERENCE = ROOT.parent/'inference'
needs_inference = pytest.mark.skipif(not INFERENCE.is_dir(), reason='inference/ sources not present on this machine')


def function_source(filename, name):
    src = Path(filename).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


@needs_inference
def test_geodata_is_byte_copy():
    assert (ROOT/'scc_geodata.py').read_bytes() == (INFERENCE/'single-so/python/geodata.py').read_bytes()


@needs_inference
@pytest.mark.parametrize('name', sorted(scc_inference.VERBATIM))
def test_functions_are_verbatim(name):
    assert function_source(ROOT/'scc_inference.py', name) == function_source(INFERENCE/scc_inference.VERBATIM[name], name)


@needs_inference
def test_tile_slicing_matches_feature_extract():
    src = (INFERENCE/'projects/SCC-Loc/Retrieval_Models/feature_extract.py').read_text(encoding='utf-8')
    assert ('img = cv2.resize(SATELLITE[int(mid_x - block_size[0] / 2):int(mid_x + block_size[0] / 2),\n'
            '                  int(mid_y - block_size[1] / 2):int(mid_y + block_size[1] / 2)], (retrieval_size,retrieval_size))') in src
    assert "side_px = max(10, 2 * int(math.ceil(side_m / (2 * ref_resolution))))" in (INFERENCE/'projects/SCC-Loc/utils.py').read_text(encoding='utf-8')


def test_block_polygon_round_trip():
    img = np.full((300, 400, 3), 100, np.uint8)
    ref, rot = scc_inference.dumpRotateImage(img, 37.)
    frame = {'view': type('V', (), {'x0': 1000., 'y0': 5000., 'res': 2.})(), 'rotation': rot, 'side_px': 60}
    corners = scc_inference.block_map_corners(frame, [200., 210.])
    back = scc_inference.map_to_rotated(frame, corners)
    np.testing.assert_allclose(back, [[180, 170], [240, 170], [240, 230], [180, 230]], atol=1e-6)


def test_tile_tensor_matches_gray_replication():
    from preprocess import tensor_image
    g = np.random.default_rng(0).integers(0, 255, (336, 336), dtype=np.uint8)
    cfg = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    a = scc_inference.tile_tensor(np.repeat(g[..., None], 3, -1), cfg['mean'], cfg['std'])
    assert np.allclose(a.numpy(), tensor_image(g, cfg).numpy())


@pytest.mark.parametrize('w,h', [(640, 512), (641, 511), (500, 500)])
def test_query_preprocessing_matches_crop_center(tmp_path, w, h):
    """Training query_image == SCC crop_center(min side) + PIL resize(336) for 8-bit IR, as the C++ path stages it."""
    import cv2
    from preprocess import query_image
    g = np.random.default_rng(w).integers(0, 255, (h, w), dtype=np.uint8)
    f = tmp_path/'q.png'; cv2.imwrite(str(f), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))   # backends stages BGR png
    side = min(w, h)
    expected = np.asarray(scc_inference.crop_center(str(f), side, side).resize((336, 336)))
    actual, _ = query_image(str(f), {'size': 336, 'intensity': {'ir': {'uint8': [0, 255], 'nodata': None}}})
    assert expected.shape == (336, 336, 3) and (expected == expected[..., :1]).all()
    assert np.array_equal(actual, expected[..., 0])


def test_search_prior_places_window_around_truth_only():
    from prepare_dataset import search_prior
    from pyproj import Transformer
    ref = type('R', (), {'forward': Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True),
                         'inverse': Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True)})()
    rec = dict(droneLongitude=87.3, droneLatitude=42.3, droneAltitude=1000., droneLongitude_gt=87.05, droneLatitude_gt=42.02,
               droneAltitude_gt=900., sensorYaw=0, sensorPitch=0, sensorRoll=0, innsYaw=0, innsPitch=-80, innsRoll=0, hfov=50, vfov=40)
    c = {'seed': 1, 'candidates': {'center_offset_m': 300, 'altitude_error_m': 100, 'train_window_m': 1200, 'eval_window_m': 2000},
         'inference': {'reference': {'window_m': 600}}}
    item = {'id': 'DS/s/1', 'record': rec, 'number': 1}
    gx, gy = ref.forward.transform(87.05, 42.02)
    for split, window in (('train', 1200), ('val', 2000), ('test', 2000)):
        t, w, src = search_prior(item, split, c, ref)
        tx, ty = ref.forward.transform(t['lon'], t['lat'])
        assert w == window and src == 'gt_offset' and np.hypot(tx-gx, ty-gy) <= 300 + 1e-6 and abs(t['altitude']-900.) <= 100
        assert search_prior(item, split, c, ref)[0] == t                       # deterministic
    rec['droneLongitude'] = 0.                                                   # the (unreliable) prior is not used
    assert search_prior(item, 'val', c, ref)[0]['lon'] == t['lon']
