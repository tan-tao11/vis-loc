"""Lean MLP config inheritance, train-candidate subsampling and the satellite intensity guard."""
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio import Affine
from types import SimpleNamespace
from common import config, ROOT
from prepare_dataset import subsample_train, satellite_intensity_check, prepare
from tests.test_sequence_split import sequence_fixture


def test_mlp_config_inherits_train_defaults():
    full, lean = config(ROOT/'configs/train.yaml'), config(ROOT/'configs/mlp.yaml')
    assert 'base' not in lean and lean['model']['gem_p'] == full['model']['gem_p']
    assert lean['labels'] == full['labels'] and lean['extract'] == full['extract']
    assert lean['split']['sequence_splits'] and lean['candidates']['train_negatives'] == 128
    assert not any(lean[s][k] for s, k in [('data', 'mapping_confirmed'), ('pose', 'calibration_confirmed'),
        ('geo', 'vertical_confirmed'), ('preprocess', 'deployment_confirmed'), ('model', 'deployed_backbone_confirmed')])


def test_subsample_keeps_positives_and_is_deterministic():
    cands = [{'id': str(i), 'label': 1 if i < 3 else 0 if i % 3 else -1, 'center_map': [float(i), 0.]} for i in range(300)]
    a = subsample_train(cands, [0., 0.], 20, 1, 'q'); b = subsample_train(cands, [0., 0.], 20, 1, 'q')
    assert [x['id'] for x in a] == [x['id'] for x in b]
    assert sum(x['label'] == 1 for x in a) == 3 and sum(x['label'] == 0 for x in a) == 20
    assert not any(x['label'] == -1 for x in a)
    nearest = sorted((x for x in cands if x['label'] == 0), key=lambda x: x['center_map'][0])[:10]
    assert all(x in a for x in nearest)
    assert [int(x['id']) for x in a] == sorted(int(x['id']) for x in a)  # grid order preserved


def test_prepare_with_train_negatives(tmp_path):
    c = sequence_fixture(tmp_path); c['candidates']['train_negatives'] = 4
    m = prepare(c)
    for g in m['groups']:
        labels = [x['label'] for x in g['candidates']]
        if g['split'] == 'train': assert labels.count(0) <= 4 and -1 not in labels and 1 in labels
        else: assert -1 in labels or labels.count(0) > 4  # val/test keep the full grid


@pytest.mark.parametrize('dtype,spec,bad', [
    ('uint16', {'uint16': [0, 255], 'nodata': None}, True),     # real range ~ thousands -> pure white
    ('uint16', {'uint16': [0, 4000], 'nodata': None}, False),
    ('float32', {'float': [0, 1], 'nodata': None}, True)])
def test_intensity_guard_flags_white_tiles(dtype, spec, bad):
    a = np.random.default_rng(0).uniform(500, 3500, (300, 300)).astype(dtype)
    with MemoryFile() as mem, mem.open(driver='GTiff', width=300, height=300, count=1, dtype=dtype, transform=Affine.identity()) as ds:
        ds.write(a, 1); raster = SimpleNamespace(ds=ds)
        if bad:
            with pytest.raises(ValueError, match='saturates'): satellite_intensity_check(raster, spec)
        else:
            assert satellite_intensity_check(raster, spec)['mapped_white_fraction'] < .1
