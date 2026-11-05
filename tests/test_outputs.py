import copy
import csv
from pathlib import Path
import numpy as np
import pytest
import torch
from common import config, ROOT, read_json, write_json
from evaluate import evaluate_arrays, rank_metrics
from diagnostics import evaluation_orders, contact_sheet
from outputs import fresh_directory, write_evaluation, write_training_history, atomic_torch_export
from synthetic import make_fixture
from validate_geometry import run


def sample():
    candidates = [dict(label=label, coverage=float(label == 1), scale_m=100., valid_fraction=1., center_map=[1,2]) for label in [1,0,-1]]
    groups = [dict(id=i, group='DS', sequence='s', protocol='p', split='val', candidates=copy.deepcopy(candidates)) for i in ['=1+2', 'b']]
    groups[1]['candidates'][0]['label'] = 0
    m = dict(manifest_id='fixture', groups=groups, total_matched=2, processed=2, failures=[], limited=False, synthetic=True)
    arrays = {i: (np.array([1.,0.], dtype=np.float32), np.array([[.1,0],[.9,0],[.2,0]],dtype=np.float32)) for i in ['=1+2','b']}
    return m, evaluate_arrays(m, arrays)


def test_report_ranking_csv_and_failed_rebuild_preserve_previous(tmp_path, monkeypatch):
    import outputs
    m, report=sample(); dest=tmp_path/'val.json'
    first=write_evaluation(dest, report, m)
    with (first/'queries.csv').open(encoding='utf-8-sig',newline='') as f: queries=list(csv.DictReader(f))
    assert queries[0]['id']=="'=1+2" and queries[0]['first_positive_rank']=='3'
    assert queries[1]['outcome']=='no_positive_candidate'
    with (first/'candidates.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert [x['candidate_index'] for x in rows[:3]]==['1','2','0']
    assert float(rows[0]['score']) == pytest.approx(.9)
    second=write_evaluation(dest, report, m)
    assert second != first and (first/'index.html').is_file()
    before=dest.read_bytes()
    def fail(*args, **kwargs): raise OSError('simulated disk failure')
    monkeypatch.setattr(outputs,'write_csv',fail)
    with pytest.raises(OSError): write_evaluation(dest,report,m)
    assert dest.read_bytes()==before
    assert any(read_json(p)['status']=='failed' for p in tmp_path.rglob('status.json'))


def test_incomplete_or_mismatched_reports_rejected(tmp_path):
    m, report=sample()
    report['results']['gem1024']['rows'].pop()
    with pytest.raises(ValueError, match='Incomplete'): write_evaluation(tmp_path/'x.json', report,m)
    with pytest.raises(ValueError, match='incomplete'): evaluation_orders(m,report)
    m, report=sample(); report['manifest_id']='wrong'
    with pytest.raises(ValueError, match='mismatch'): evaluation_orders(m,report)


def test_lora_and_mlp_modes_require_valid_permutation():
    m, report=sample()
    report['results']['lora_gem1024']=report['results'].pop('gem1024')
    assert evaluation_orders(m,report)[0]=='lora_gem1024'
    report['results']['mlp512']=copy.deepcopy(report['results']['lora_gem1024'])
    assert evaluation_orders(m,report,'mlp512')[0]=='mlp512'
    report['results']['mlp512']['rows'][0]['order']=[0,0,2]
    with pytest.raises(ValueError,match='order'): evaluation_orders(m,report,'mlp512')


def test_bad_scores_and_empty_split_fail():
    with pytest.raises(ValueError): rank_metrics([[1,2]],[1,0])
    with pytest.raises(ValueError): rank_metrics([1],[1,0])
    with pytest.raises(ValueError): rank_metrics([float('nan')],[1])
    m,_=sample()
    with pytest.raises(ValueError,match='Empty'): evaluate_arrays(m,{},split='test')


def test_existing_output_is_preserved(tmp_path):
    (tmp_path/'keep.txt').write_text('user data')
    with pytest.raises(ValueError,match='NEW'): fresh_directory(tmp_path)
    assert (tmp_path/'keep.txt').read_text()=='user data'


def test_evaluation_attempt_does_not_mislabel_previous_result(tmp_path):
    from outputs import evaluation_attempt
    dest=tmp_path/'val.json'; write_json(dest,{'old':'success'}); before=dest.read_bytes()
    with pytest.raises(RuntimeError,match='OOM'):
        with evaluation_attempt(dest): raise RuntimeError('simulated OOM')
    assert dest.read_bytes()==before
    assert read_json(str(dest)+'.attempt.json')['status']=='failed'
    assert 'OOM' in read_json(str(dest)+'.attempt.json')['traceback']


def test_export_failed_write_preserves_existing_weight(tmp_path, monkeypatch):
    dest=tmp_path/'model.pth'; payload={'state_dict':{'a':torch.tensor([1.])}}
    atomic_torch_export(dest,payload); before=dest.read_bytes()
    def fail(obj, file):
        Path(file).write_bytes(b'partial'); raise OSError('disk full')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError): atomic_torch_export(dest,payload)
    assert dest.read_bytes()==before and not list(tmp_path.glob('*.partial'))


def test_history_handles_missing_loss_and_single_epoch(tmp_path):
    write_training_history(tmp_path,[dict(epoch=0,loss=None,val_recall_at_1_all=.5,gradient_norm_sum={'A':0.})])
    from PIL import Image
    assert Image.open(tmp_path/'history.png').size==(1100,660)
    assert (tmp_path/'history.csv').read_text(encoding='utf-8-sig').startswith('epoch,loss')


def test_invalid_controls_produce_failed_readable_report(tmp_path):
    c=config(ROOT/'configs/train.yaml'); c['data']['root']=str(tmp_path/'missing')
    controls=tmp_path/'bad.csv'; controls.write_text('wrong,headers\n')
    report=run(c,tmp_path/'diagnostic',controls_file=controls)
    assert report['diagnostic_status']=='failed'
    assert any(i['field']=='controls_csv' for i in report['issues'])
    assert (tmp_path/'diagnostic/index.html').is_file()


def test_geometry_per_frame_control_failure_and_no_control_status(tmp_path, monkeypatch):
    import validate_geometry as validator
    c=config(make_fixture(tmp_path/'fixture')); original=validator.inspect_frame
    def inspect(*args,**kwargs):
        row=original(*args,**kwargs)
        if row['group']=='DS':
            row['controls']=[dict(id=row['id'],role='check',line=2,error_m=31.,error_east_m=31.,error_north_m=0.,
                longitude=row['camera_lon_lat_normal'][0],latitude=row['camera_lon_lat_normal'][1])]
        return row
    monkeypatch.setattr(validator,'inspect_frame',inspect)
    report=run(c,tmp_path/'geometry',max_frames=4,tolerance=1.)
    assert report['diagnostic_status']=='failed'
    assert report['frames'][0]['diagnostic_status']=='control_out_of_tolerance'
    assert report['frames'][1]['diagnostic_status']=='footprint_computable_unverified'
    assert 'control_out_of_tolerance' in (tmp_path/'geometry/index.html').read_text(encoding='utf-8')
