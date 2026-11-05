import copy
from pathlib import Path
import pytest
from common import config, verify_manifest
from prepare_dataset import prepare, sequence_assignments
from records import scan
from synthetic import make_fixture


def test_sequence_mapping_is_global_and_order_independent():
    pairs = [{'group': g, 'sequence': s} for g in ('DS', 'DV', 'NS', 'NV') for s in ('sequence1', 'sequence2')]
    split = {'sequence_splits': {'sequence1': 'train', 'sequence2': 'val'}}
    assert sequence_assignments(pairs, split) == sequence_assignments(list(reversed(pairs)), split)
    assert sequence_assignments(pairs, split) == split['sequence_splits']


@pytest.mark.parametrize('mapping', [None, {}, {'sequence1': 'validation'}, {1: 'train'}, {'sequence2': 'train'}])
def test_missing_or_invalid_assignments_rejected(mapping):
    with pytest.raises(ValueError):
        sequence_assignments([{'sequence': 'sequence1'}], {'sequence_splits': mapping})


def sequence_fixture(tmp_path):
    c = config(make_fixture(tmp_path/'fixture'))
    Path(c['split']['regions_geojson']).unlink()
    c['split'].update(mode='sequence', regions_geojson=None,
        sequence_splits={'sequence_train': 'train', 'sequence_val': 'val', 'sequence_test': 'test'})
    return c


def test_prepare_without_regions_and_limit_keeps_assignments(tmp_path):
    c = sequence_fixture(tmp_path)
    # Deliberately overlapping search buffers: sequence mode must report, not conceal, this.
    c['split']['buffer_m'] = 10000
    full = prepare(c)
    verify_manifest(full)
    assert not full['failures'] and not full['leakage']
    assert len(full['groups']) == 12
    assert any(x['reason'] == 'candidate_geographic_leakage' for x in full['split_warnings'])
    for g in full['groups']:
        assert g['split'] == c['split']['sequence_splits'][g['sequence']]
    limited_config = copy.deepcopy(c)
    limited_config['output'] = str(tmp_path/'limited')
    limited = prepare(limited_config, limit=2)
    assert limited['sequence_splits'] == full['sequence_splits']
    by_id = {g['id']: g['split'] for g in full['groups']}
    assert all(g['split'] == by_id[g['id']] for g in limited['groups'])


def test_sequence_mode_still_rejects_duplicate_images_across_splits(tmp_path):
    c = sequence_fixture(tmp_path)
    pairs = scan(c)['pairs']
    first = next(p for p in pairs if p['group'] == 'DV' and p['sequence'] == 'sequence_train')
    second = next(p for p in pairs if p['group'] == 'DV' and p['sequence'] == 'sequence_val')
    Path(second['image']).write_bytes(Path(first['image']).read_bytes())
    with pytest.raises(ValueError, match='split leakage'):
        prepare(c)
