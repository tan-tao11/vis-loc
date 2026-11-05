import copy
import hashlib
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from common import ROOT,config,read_json
from synthetic import make_fixture
from prepare_dataset import prepare
from lora import LoRARetriever
from ddp_runtime import shard_window, merge_evaluation, validate_devices
from train_lora_ddp import validate_settings,launch
from tests.test_lora import TinyDino


class TinyImages:
    def __init__(self,m): self.m=m
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,g,index=None):
        seed=int(hashlib.sha256((g['id']+str(index)).encode()).hexdigest()[:8],16)
        return torch.randn(5,1024,generator=torch.Generator().manual_seed(seed))


def tiny_factory(m,c,device):
    return LoRARetriever(TinyDino(),c['lora']).to(device),{'synthetic_test_backbone':True},TinyImages(m)


class FailingImages(TinyImages):
    def __init__(self,m,model): super().__init__(m); self.model=model
    def read(self,g,index=None):
        import torch.distributed as dist
        if dist.get_rank()==1 and self.model.training: raise OSError('injected rank-one read failure')
        return super().read(g,index)


def failing_factory(m,c,device):
    model=LoRARetriever(TinyDino(),c['lora']).to(device)
    return model,{'synthetic_test_backbone':True},FailingImages(m,model)


@pytest.mark.parametrize('count',[1,2,3,4,5,7])
def test_tail_shards_have_equal_steps_without_duplicate_queries(count):
    rows=list(range(count)); shards=[shard_window(rows,r,2) for r in range(2)]
    assert len(shards[0])==len(shards[1])
    assert sorted(x for s in shards for x in s if x is not None)==rows


def test_production_cannot_mislabel_shared_gpu_or_cpu():
    with pytest.raises(ValueError,match='DISTINCT'): validate_devices(['cuda:0','cuda:0'])
    with pytest.raises(ValueError,match='DISTINCT'): validate_devices(['cpu','cpu'])
    c=config(ROOT/'configs/lora_ddp.yaml'); c['train']['global_batch_groups']=3
    with pytest.raises(ValueError,match='multiple'): validate_settings(c,2)


def test_distributed_evaluation_rejects_duplicates_and_counts_uncovered():
    from evaluate import rank_metrics
    groups=[{'id':str(i),'split':'val','group':'DS','sequence':'s','protocol':'p',
        'candidates':[{'label':1 if i==0 else 0,'scale_m':100}]} for i in range(2)]
    m={'groups':groups,'manifest_id':'fixture','synthetic':True,'total_matched':2,'processed':2,'failures':[],'limited':False}
    parts=[{'rows':[dict(id=str(i),scores=[.8],**rank_metrics([.8],[groups[i]['candidates'][0]['label']]))],'seconds':1.} for i in range(2)]
    result=merge_evaluation(m,parts,'val')['results']['lora_gem1024']['overall']
    assert result['queries']==2 and result['recall_all_legal']['1']==.5
    with pytest.raises(ValueError,match='duplicating'): merge_evaluation(m,[parts[0],parts[0]],'val')


def test_cpu_ddp_gradient_average(tmp_path):
    from ddp_check import check
    report=check(tmp_path/'probe',['cpu','cpu'],diagnostic=True)
    assert report['status']=='passed' and not report['distinct_gpu_test_passed']


class ScalarLoss(torch.nn.Module):
    def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(0.))
    def forward(self,item): return self.weight*0 if item is None else (self.weight-float(item)).square()


def tail_gradient_worker(rank,devices,uri,timeout,output):
    import torch.distributed as dist
    from ddp_runtime import init_group,wrap_ddp,backward_window
    from common import write_json
    init_group(rank,devices,uri,timeout)
    try:
        model=ScalarLoss(); ddp=wrap_ddp(model,'cpu'); scaler=torch.amp.GradScaler('cuda',enabled=False)
        results=[]
        for count in (1,3,4,5,7):
            model.zero_grad(set_to_none=True)
            backward_window(ddp,shard_window(list(range(1,count+1)),rank,2),count,scaler)
            expected=-(count+1.)
            assert model.weight.grad.item()==pytest.approx(expected,abs=1e-6)
            results.append(dict(count=count,gradient=model.weight.grad.item(),expected=expected))
        write_json(Path(output)/f'rank_{rank}.json',results)
    finally: dist.destroy_process_group()


def test_ddp_accumulation_tail_equals_serial_global_mean(tmp_path):
    from ddp_runtime import spawn_workers
    spawn_workers(tail_gradient_worker,['cpu','cpu'],tmp_path,(str(tmp_path),),30)
    assert read_json(tmp_path/'rank_0.json')==read_json(tmp_path/'rank_1.json')


def fixture_settings(tmp_path):
    c=config(make_fixture(tmp_path/'data')); prepare(c)
    manifest=Path(c['output'])/'manifest.json'
    settings=config(ROOT/'configs/lora_ddp.yaml')
    settings['lora'].update(last_n_blocks=1,rank=2,alpha=4.,dropout=.1)
    settings['train'].update(epochs=3,patience=4,amp=False,max_positives=1,hard_negatives=1,random_negatives=1,global_batch_groups=4)
    settings['evaluation']['image_batch_size']=4; settings['distributed']['timeout_seconds']=60
    return manifest,settings


def test_two_process_training_resume_restores_rank_rng_and_identical_adapters(tmp_path):
    manifest,c=fixture_settings(tmp_path)
    launch(c,manifest,tmp_path/'full',['cpu','cpu'],diagnostic=True,fixture=tiny_factory)
    launch(c,manifest,tmp_path/'resumed',['cpu','cpu'],diagnostic=True,stop_after_epoch=0,fixture=tiny_factory)
    assert read_json(tmp_path/'resumed/job_status.json')['status']=='paused'
    launch(c,manifest,tmp_path/'resumed',['cpu','cpu'],resume=tmp_path/'resumed/last.pth',diagnostic=True,fixture=tiny_factory)
    full=torch.load(tmp_path/'full/last.pth',weights_only=True)
    resumed=torch.load(tmp_path/'resumed/last.pth',weights_only=True)
    assert full['history']==resumed['history']
    assert all(torch.equal(v,resumed['state_dict'][k]) for k,v in full['state_dict'].items())
    assert full['scheduler']==resumed['scheduler']
    assert all(len(set(r['replica_adapter_hashes']))==1 for r in full['history'])
    assert len(full['rng_by_rank'])==2
    report=read_json(tmp_path/'full/best_validation.json')
    assert report['results']['lora_gem1024']['overall']['queries']==4
    assert not report['evaluation_protocol']['query_padding']
    assert read_json(tmp_path/'full/job_status.json')['status']=='complete'


def test_one_rank_failure_stops_job_and_preserves_failure_report(tmp_path):
    manifest,c=fixture_settings(tmp_path)
    with pytest.raises(Exception): launch(c,manifest,tmp_path/'failed',['cpu','cpu'],diagnostic=True,fixture=failing_factory)
    assert read_json(tmp_path/'failed/job_status.json')['status']=='failed'
    assert not (tmp_path/'failed/best.pth').exists()
    assert 'injected rank-one read failure' in (tmp_path/'failed/rank_1_failure.json').read_text()
