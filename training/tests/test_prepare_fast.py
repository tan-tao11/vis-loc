"""Compare preparation fast path with full crop and raster polygonization."""
from types import SimpleNamespace
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio import Affine
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from preprocess import local_rotated_crop, rotation_affine, valid_polygon
from prepare_dataset import label


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'float32'])
@pytest.mark.parametrize('angle', [0, 37, 90])
def test_mask_only_matches_full_crop(dtype, angle):
    a = np.full((200, 240), 100, dtype=dtype)
    a[80:110, 90:125] = 0
    if dtype == 'float32': a[50:55, 50:60] = np.nan
    spec = {'uint8': [0, 255], 'uint16': [0, 65535], 'float': [0, 200], 'nodata': 0}
    with MemoryFile() as mem:
        with mem.open(driver='GTiff', width=240, height=200, count=1,
                      dtype=dtype, transform=Affine.translation(100, 200), nodata=0) as ds:
            ds.write(a, 1)
            raster = SimpleNamespace(ds=ds, width=240, height=200)
            inv, _ = rotation_affine(240, 200, angle)
            for box in ([60, 60, 140, 140], [-10, -10, 70, 70], [400, 400, 480, 480]):
                _, expected, corners = local_rotated_crop(raster, inv, box, spec)
                image, actual, got_corners = local_rotated_crop(raster, inv, box, spec, mask_only=True)
                assert image is None
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(got_corners, corners)


@pytest.mark.parametrize('angle', [0, 37, 90])
@pytest.mark.parametrize('kind', ['full', 'hole', 'empty'])
def test_polygon_and_labels_match_raster_path(angle, kind):
    mask = np.ones((40, 60), dtype=bool)
    if kind == 'hole': mask[10:20, 20:30] = False
    if kind == 'empty': mask[:] = False
    inv, _ = rotation_affine(240, 200, angle)
    box = [20, 30, 80, 70]
    transform = Affine(2, .1, 1000, .2, -2, 2000)
    tf = transform * Affine(*inv[:2].ravel()) * Affine.translation(*box[:2])
    expected = unary_union([shape(g) for g, v in shapes(mask.astype('uint8'), mask=mask, transform=tf) if v == 1])
    actual = valid_polygon(mask, inv, box, transform)
    assert actual.equals(expected)
    cfg = dict(positive_coverage=.8, negative_coverage=.05, max_area_ratio=4., max_center_distance_ratio=1.)
    query = shape({'type': 'Polygon', 'coordinates': [[tf * p for p in [(5, 5), (5, 30), (40, 30), (40, 5), (5, 5)]]]})
    assert label(query, actual, cfg) == pytest.approx(label(query, expected, cfg))
