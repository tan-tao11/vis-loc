import copy
import numpy as np
import pytest
from shapely.geometry import box, mapping
from common import config, ROOT, require_real_ready, verify_manifest
from records import parse_records, scan
from prepare_dataset import label, split_audit, prepare
from scc_inference import telemetry
from synthetic import make_fixture

def test_alias_and_multiple_records(tmp_path):
    p = tmp_path/'r.txt'; p.write_text('number=1\nsequence=s1\ndroneLongtitude=87\ndroneLongitude=87\nccdSize=12\nnumber=2\nsequence=s2\n')
    rs, fmt = parse_records(p); assert len(rs) == 2 and rs[0]['droneLongitude'] == 87 and rs[0]['ccdsize'] == 12
    p.write_text('number=1\ndroneLongtitude=87\ndroneLongitude=88')
    with pytest.raises(ValueError, match='conflict'): parse_records(p)

def test_labels_and_leakage():
    c = config(ROOT/'configs/train.yaml'); q = box(0, 0, 10, 10)
    assert label(q, box(-2.5, -2.5, 12.5, 12.5), c['labels'])[0] == 1
    assert label(q, box(0, 0, 6, 10), c['labels'])[0] == -1
    assert label(q, box(20, 0, 30, 10), c['labels'])[0] == 0
    assert label(q, box(-100, -100, 100, 100), c['labels'])[0] == -1
    gs = [{'group': 'DS', 'sequence': str(i), 'split': s, 'search_footprint': mapping(q)} for i, s in enumerate(('train', 'val'))]
    assert split_audit(gs, 1)[0]['reason'] == 'candidate_geographic_leakage'

def test_real_guard_and_no_gt_prior():
    c = config(ROOT/'configs/train.yaml')
    with pytest.raises(ValueError, match='blocked'): require_real_ready(c)
    r = dict(droneLongitude=87., droneLatitude=42., droneAltitude=2000., droneLongitude_gt=90., droneLatitude_gt=40.,
             sensorYaw=0, sensorPitch=0, sensorRoll=0, innsYaw=1, innsPitch=-80, innsRoll=0, hfov=50, vfov=40)
    t1 = telemetry(r); r.update(droneLongitude_gt=0., droneLatitude_gt=0.)
    assert telemetry(r) == t1 and t1['lon'] == 87. and t1['lat'] == 42.  # search uses the prior only

def test_synthetic_mapping_preparation(tmp_path):
    cfgfile = make_fixture(tmp_path); c = config(cfgfile); report = scan(c)
    assert not report['errors']; assert len(report['pairs']) == 12
    # Repeated frame number 1000 in three sequences must remain distinct.
    assert len({r['id'] for r in report['pairs']}) == 12
    m = prepare(c); assert not m['failures']; assert not m['leakage']; assert len(m['groups']) == 12
    assert all(any(x['label'] == 1 for x in g['candidates']) for g in m['groups'])
    assert all(not x['training_gt_inserted'] for g in m['groups'] for x in g['candidates'])
    verify_manifest(m)
    corrupt = copy.deepcopy(m); corrupt['groups'][0]['number'] = 999999
    with pytest.raises(ValueError, match='integrity'): verify_manifest(corrupt)
    # No sequence in a group-wide file is an error, not an inferred ordering.
    p = tmp_path/'data/day_IR_slant.txt'; txt = p.read_text(); p.write_text(txt.replace('sequence,', '').replace('sequence_train,', '').replace('sequence_val,', '').replace('sequence_test,', ''))
    assert any(e['reason'] == 'record_error' for e in scan(c)['errors'])
