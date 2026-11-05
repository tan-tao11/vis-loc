import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from common import config, read_json, write_json, digest
from synthetic import make_fixture
from prepare_dataset import prepare
from train_mlp_ddp import launch, MLPGroupLoss
from models import RetrievalHeads


def tiny_cache(m,cache_dir):
    rng=np.random.default_rng(123)
    arrays={}
    for g in m['groups']:
        q=rng.normal(size=1024).astype('float32')
        s=rng.normal(size=(len(g['candidates']),1024)).astype('float32')
        arrays[g['id']]=(q,s)
    return arrays,{'fingerprint':digest({'fixture':True,'manifest':m['manifest_id']}),
        'backbone':{'checkpoint_sha256':'synthetic_test_backbone'}}


def failing_cache(m,cache_dir):
    import torch.distributed as dist
    arrays,index=tiny_cache(m,cache_dir)
    if dist.get_rank()==1:
        for g in m['groups']:
            if g['split']=='train': arrays[g['id']][0][:]=np.nan
    return arrays,index


def fixture_manifest(tmp_path):
    c=config(make_fixture(tmp_path/'data'))
    c['model'].update(hidden_dim=16,dropout=.1)
    c['train'].update(epochs=3,patience=4,batch_size=3,hard_negative_count=2)
    prepare(c)
    return Path(c['output'])/'manifest.json'


def head_gradient_worker(rank,devices,uri,timeout,output):
    import torch.distributed as dist
    from ddp_runtime import init_group,wrap_ddp,shard_window,backward_window
    init_group(rank,devices,uri,timeout)
    try:
        torch.set_num_threads(2); torch.manual_seed(13)
        model=RetrievalHeads(hidden_dim=16)
        groups=[{'id':str(i),'candidates':[{'label':1},{'label':0},{'label':-1}]} for i in range(5)]
        arrays,_=tiny_cache({'groups':groups,'manifest_id':'gradient'},'')
        serial=MLPGroupLoss(copy.deepcopy(model),arrays,'cpu',.07)
        ddp=wrap_ddp(MLPGroupLoss(model,arrays,'cpu',.07),'cpu')
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        errors=[]
        for count in (1,3,4,5):
            items=[(g,[0,1,2]) for g in groups[:count]]
            serial.zero_grad(set_to_none=True); ddp.zero_grad(set_to_none=True)
            torch.stack([serial(item) for item in items]).mean().backward()
            backward_window(ddp,shard_window(items,rank,2),count,scaler)
            error=0.
            for actual,expected in zip(model.parameters(),serial.heads.parameters()):
                assert actual.grad is not None and expected.grad is not None
                error=max(error,float((actual.grad-expected.grad).abs().max()))
                torch.testing.assert_close(actual.grad,expected.grad,rtol=1e-5,atol=2e-6)
            errors.append({'groups':count,'max_gradient_error':error})
        write_json(Path(output)/f'rank_{rank}.json',errors)
    finally: dist.destroy_process_group()


def test_actual_two_head_ddp_gradients_equal_serial_global_mean(tmp_path):
    from ddp_runtime import spawn_workers
    spawn_workers(head_gradient_worker,['cpu','cpu'],tmp_path,(str(tmp_path),),60)
    assert read_json(tmp_path/'rank_0.json')==read_json(tmp_path/'rank_1.json')


def test_training_resume_and_existing_export_adapter(tmp_path):
    from export import export
    from evaluate import load_heads_checkpoint
    from integration.retrieval_adapter import RetrievalAdapter
    manifest=fixture_manifest(tmp_path)
    launch(manifest,'unused',tmp_path/'full',['cpu','cpu'],diagnostic=True,fixture=tiny_cache)
    launch(manifest,'unused',tmp_path/'resume',['cpu','cpu'],diagnostic=True,stop_after_epoch=0,fixture=tiny_cache)
    assert read_json(tmp_path/'resume/job_status.json')['status']=='paused'
    launch(manifest,'unused',tmp_path/'resume',['cpu','cpu'],resume=tmp_path/'resume/last.pth',diagnostic=True,fixture=tiny_cache)
    full=torch.load(tmp_path/'full/last.pth',weights_only=True)
    resumed=torch.load(tmp_path/'resume/last.pth',weights_only=True)
    assert full['history']==resumed['history']
    assert full['scheduler']==resumed['scheduler']
    assert all(torch.equal(value,resumed['state_dict'][name]) for name,value in full['state_dict'].items())
    for row in full['history']:
        assert len(set(row['replica_head_hashes']))==1
        assert all(v>0 for v in row['gradient_norm_sum'].values())
        assert row['query_groups']==4 and row['optimizer_updates']==2  # Last window has ONE real group.
    report=read_json(tmp_path/'full/best_validation.json')
    for result in report['results'].values(): assert result['overall']['queries']==4
    assert read_json(tmp_path/'resume/job_status.json')['status']=='complete'
    destination=tmp_path/'head.pth'; metadata=export(tmp_path/'full/best.pth',destination)
    adapter=RetrievalAdapter(destination,True,expected_backbone_hash=metadata['backbone']['checkpoint_sha256'],
        expected_preprocess_pose=metadata['preprocess_pose_fingerprint'],allow_synthetic=True)
    heads,_=load_heads_checkpoint(tmp_path/'full/best.pth')
    x=torch.randn(2,1024)
    for domain in ('ir','satellite'): torch.testing.assert_close(adapter.apply(x,domain),heads(x,domain),rtol=0,atol=0)


def test_failed_rank_aborts_without_exportable_best(tmp_path):
    manifest=fixture_manifest(tmp_path)
    with pytest.raises(Exception): launch(manifest,'unused',tmp_path/'failed',['cpu','cpu'],diagnostic=True,fixture=failing_cache)
    assert read_json(tmp_path/'failed/job_status.json')['status']=='failed'
    assert not (tmp_path/'failed/best.pth').exists()
    assert 'Nonfinite contrastive loss' in (tmp_path/'failed/rank_1_failure.json').read_text()
