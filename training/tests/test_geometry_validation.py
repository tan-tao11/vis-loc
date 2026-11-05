import copy
import numpy as np
import pytest
from common import ROOT, config
from geometry import CameraFrame, Ground, camera_rotation, intrinsics
from synthetic import make_fixture
from validate_geometry import (GEOD, analytic_checks, static_checks, control_error,
    summarize_controls, load_controls, run)


def test_independent_analytic_geometry_contracts():
    assert all(analytic_checks().values())


def test_improper_rotation_and_eo_optical_axis_rejected():
    c = config(ROOT/'configs/train.yaml')
    c['pose']['sensor_from_cam'] = np.diag([1, 1, -1]).tolist()
    c['pose']['eo_inns_from_cam'] = np.eye(3).tolist()
    errors = {x['field'] for x in static_checks(c) if x['level'] == 'error'}
    assert errors == {'pose.sensor_from_cam', 'pose.eo_inns_from_cam'}


def test_independent_landmark_detects_half_turn_missed_by_center_only():
    c = config(ROOT/'configs/train.yaml')
    r = dict(innsYaw=0, innsPitch=0, innsRoll=0, sensorYaw=0, sensorPitch=0, sensorRoll=0, isEO=0, hfov=90, vfov=90)
    R = camera_rotation(r, c['pose']); K = intrinsics(200, 200, r)
    ground = Ground({'dsm': None, 'flat_fallback': True, 'flat_ground_normal_m': 100})
    frame = CameraFrame(87, 42, 200, 0)
    lon, lat, _ = GEOD.fwd(87, 42, 90, 50)  # Analytical: 100 m AGL, x/f=0.5 => 50 m east.
    point = {'id': 'DV/sequence1/1', 'line': 2, 'pixel_x': 150, 'pixel_y': 100, 'longitude': lon, 'latitude': lat, 'role': 'check'}
    item = {'width': 200, 'height': 200}
    good = control_error(point, item, R, K, frame, ground, c['geo'])
    wrong = control_error(point, item, R @ np.diag([-1, -1, 1]), K, frame, ground, c['geo'])
    assert good['error_m'] < .02
    assert wrong['error_m'] > 99.9
    center = dict(point, pixel_x=100, longitude=87, latitude=42)
    assert control_error(center, item, R @ np.diag([-1, -1, 1]), K, frame, ground, c['geo'])['error_m'] < .001
    outside = control_error(dict(point, pixel_x=336), item, R, K, frame, ground, c['geo'])
    assert 'error' in outside
    summary = summarize_controls([good, outside], 1)
    assert summary['evaluated'] == 1 and summary['failed_or_unmatched'] == 1
    assert summary['all_within_tolerance'] is False


@pytest.mark.parametrize('body', [
    'a,1,2,87,42,test\n',
    'a,1,2,87,42,check\na,1,2,87,42,calibration\n',
    'a,1,2,187,42,check\n'])
def test_control_csv_rejects_invalid_or_duplicate_points(tmp_path, body):
    filename = tmp_path/'controls.csv'
    filename.write_text('id,pixel_x,pixel_y,longitude,latitude,role\n'+body, encoding='utf-8')
    with pytest.raises(ValueError): load_controls(filename)


def test_missing_data_report_does_not_invent_datum_or_confirm(tmp_path):
    c = config(ROOT/'configs/train.yaml'); c['data']['root'] = str(tmp_path/'absent')
    before = copy.deepcopy(c)
    report = run(c, tmp_path/'report')
    assert c == before and report['summary']['evaluated_frames'] == 0
    assert any(x['field'] == 'geo.normal_to_ellipsoid_m' for x in report['issues'])
    assert report['calibration_confirmed_by_this_tool'] is False
    assert (tmp_path/'report/index.html').is_file()
    assert not (tmp_path/'report/manifest.json').exists()


def test_synthetic_diagnostic_without_confirmation_or_training_manifest(tmp_path):
    c = config(make_fixture(tmp_path/'fixture')); before = copy.deepcopy(c)
    report = run(c, tmp_path/'report', max_frames=4)
    assert report['summary']['matched_frames'] == 12
    assert report['summary']['frames_with_full_footprint'] == 4
    assert report['summary']['frames_with_errors'] == 0
    assert {r['group'] for r in report['frames']} == {'DS', 'DV', 'NS', 'NV'}
    assert all((tmp_path/'report'/r['overlay']).is_file() for r in report['frames'])
    assert c == before and not c['pose']['calibration_confirmed']
    assert not report['summary']['real_geometry_validated']
    assert not (tmp_path/'report/manifest.json').exists()
