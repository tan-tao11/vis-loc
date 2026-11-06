"""Lean MLP config inheritance, train-candidate subsampling and the satellite intensity guard."""
import numpy as np
import pytest
from rasterio import Affine
from common import config, ROOT
from prepare_dataset import subsample_train, satellite_intensity_check, prepare
from tests.test_sequence_split import sequence_fixture


def test_slant_config_inherits_train_defaults():
    full, lean = config(ROOT/'configs/train.yaml'), config(ROOT/'configs/slant.yaml')
    assert 'base' not in lean and lean['model']['gem_p'] == full['model']['gem_p']
    assert lean['labels'] == dict(full['labels'], max_area_ratio=8.0) and lean['extract'] == dict(full['extract'], batch_size=32)
    assert lean['split']['sequence_splits'] and lean['candidates']['train_negatives'] == 128
    assert lean['inference']['reference']['georeference'] == 'auto' and lean['inference']['camera']['reference_window_scale'] == 1.2
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


@pytest.mark.parametrize('dtype,value_range,bad', [
    ('uint16', [0, 255], True),      # real range ~ thousands -> pure white tiles
    ('uint16', [0, 4000], False),
    ('float32', [0, 1], True),
    ('uint16', None, 'needs')])      # deployed GeoReference refuses non-uint8 without a range
def test_intensity_guard_flags_white_tiles(tmp_path, dtype, value_range, bad):
    import rasterio
    a = np.random.default_rng(0).uniform(500, 3500, (300, 300)).astype(dtype)
    f = tmp_path/'s.tif'
    with rasterio.open(f, 'w', driver='GTiff', width=300, height=300, count=1, dtype=dtype,
                       transform=Affine(1, 0, 0, 0, -1, 300), crs='EPSG:32645') as ds: ds.write(a, 1)
    ref = {'image': str(f), 'image_bands': [1, 1, 1], 'image_value_range': value_range}
    if bad == 'needs':
        with pytest.raises(ValueError, match='image_value_range'): satellite_intensity_check(ref)
    elif bad:
        with pytest.raises(ValueError, match='saturates'): satellite_intensity_check(ref)
    else:
        assert satellite_intensity_check(ref)['mapped_white_fraction'] < .1


def test_slant_config_is_oblique_only():
    from scc_inference import optical_pose, camera_for
    c = config(ROOT/'configs/slant.yaml')
    assert set(c['data']['groups']) == {'DS', 'NS'} and all(g['isSlant'] == 1 for g in c['data']['groups'].values())
    cam = camera_for(c, 'DS')
    assert cam['oblique'] and not cam['nadir_reference_alignment'] and cam['query_crop_scale'] == c['preprocess']['query_crop_scale']
    t = dict(inns_yaw=27.6749821, inns_pitch=-15.3483686, inns_roll=-0.3845332)   # bice.md DS frame
    yaw, elevation = optical_pose(t, True, cam)
    assert abs(yaw - 27.6749821) < 1e-6 and abs(elevation + 15.3483686) < 1e-6


def test_small_gpu_is_refused(monkeypatch):
    import torch
    from types import SimpleNamespace
    import common
    gpus = [SimpleNamespace(name='NVIDIA RTX A4000', total_memory=16*2**30), SimpleNamespace(name='Quadro P400', total_memory=2*2**30),
            SimpleNamespace(name='NVIDIA RTX A4000', total_memory=16*2**30)]
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: len(gpus))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda i: gpus[i])
    assert common.usable_cuda_devices() == [0, 2]
    common.require_usable_device('cuda:0'); common.require_usable_device('cpu')
    with pytest.raises(RuntimeError, match="P400.*cuda:0', 'cuda:2"): common.require_usable_device('cuda:1')
    with pytest.raises(RuntimeError, match='only 3'): common.require_usable_device('cuda:5')


def test_parallel_prepare_matches_serial(tmp_path):
    from synthetic import make_fixture
    c1 = config(make_fixture(tmp_path/'a')); c1['output'] = str(tmp_path/'r1')
    c2 = config(make_fixture(tmp_path/'a')); c2['output'] = str(tmp_path/'r2')
    m1 = prepare(c1, workers=1); m2 = prepare(c2, workers=2)
    assert m1['groups'] == m2['groups'] and m1['failures'] == m2['failures']
