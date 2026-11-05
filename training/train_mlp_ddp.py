"""Optional two-GPU MLP training over the existing frozen-DINO feature cache."""
import argparse
from collections import defaultdict
import contextlib
from datetime import datetime, timezone
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from common import ROOT, read_json, write_json, verify_manifest, require_real_ready, digest, sha256, seed_all
from cache import load_cache
from models import create_heads
from losses import grouped_loss
from train import mine_hard_negatives
from train_lora import atomic_save, state_fingerprint
from evaluate import evaluate_arrays
from outputs import fresh_directory, write_training_history, write_evaluation
from ddp_runtime import gather, phase, init_group, validate_devices, spawn_workers, shard_window, wrap_ddp, backward_window


def read_manifest(filename):
    m=read_json(filename); verify_manifest(m); require_real_ready(m['config'])
    if m['leakage']: raise ValueError('Manifest contains split leakage')
    return m


def validate_settings(c):
    tc=c['train']
    if tc['loss']!='grouped_multi_positive' or tc['scheduler']!='cosine':
        raise ValueError('MLP DDP requires the existing grouped loss and cosine scheduler')
    for key in ('batch_size','epochs','patience','hard_refresh_epochs','hard_negative_count'):
        if type(tc[key]) is not int or tc[key]<1: raise ValueError('Invalid '+key)
    for key in ('lr','temperature'):
        if not np.isfinite(tc[key]) or tc[key]<=0: raise ValueError('Invalid '+key)
    if not np.isfinite(tc['weight_decay']) or tc['weight_decay']<0: raise ValueError('Invalid weight_decay')


class MLPGroupLoss(nn.Module):
    def __init__(self,heads,arrays,device,temperature):
        super().__init__(); self.heads=heads; self.arrays=arrays
        self.device=device; self.temperature=temperature

    def forward(self,item):
        if item is None: return sum(p.reshape(-1)[0]*0 for p in self.heads.parameters())
        group,selected=item; q,s=self.arrays[group['id']]
        q=self.heads(torch.from_numpy(q).to(self.device),'ir')
        s=self.heads(torch.from_numpy(s[selected]).to(self.device),'satellite')
        labels=torch.tensor([group['candidates'][i]['label'] for i in selected],device=self.device)
        loss,count=grouped_loss(q,s,labels,self.temperature)
        if count!=1 or not torch.isfinite(loss): raise ValueError('Invalid grouped training loss: '+group['id'])
        return loss


def schedule_groups(groups,rng):
    # Match the original MLP's domain-balanced sampling WITH replacement.
    buckets=defaultdict(list)
    for g in groups: buckets[g['group']].append(g)
    schedule=[]
    for _ in range(max(len(v) for v in buckets.values())):
        schedule.extend(rng.choice(v) for v in buckets.values())
    rng.shuffle(schedule)
    return schedule


def identity(m,index,devices,diagnostic,fixture):
    names=('train_mlp_ddp.py','ddp_runtime.py','train.py','train_lora.py','models.py','losses.py','cache.py','evaluate.py','outputs.py')
    return {'method':'mlp_ddp_v1','manifest_id':m['manifest_id'],'config':digest(m['config']),
        'cache_fingerprint':index['fingerprint'],'backbone':index['backbone'],'devices':devices,
        'world_size':len(devices),'diagnostic_only':diagnostic,'torch_version':str(torch.__version__),
        'fixture_loader':None if fixture is None else fixture.__module__+'.'+fixture.__qualname__,
        'code':{name:sha256(ROOT/name) for name in names}}


def _train(rank,devices,m,cache_dir,out,resume,diagnostic,stop_after_epoch,fixture):
    device=devices[rank]; world=len(devices); c=m['config']; tc=c['train']
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    seed_all(c['seed'])
    def load():
        if fixture: return fixture(m,cache_dir)
        _,arrays,index=load_cache(m,cache_dir)
        return arrays,index
    arrays,index=phase('verify and load feature cache',load)
    run_identity=identity(m,index,devices,diagnostic,fixture)
    if len(set(gather(digest(run_identity))))!=1: raise ValueError('Rank cache/config identities differ')
    groups=[g for g in m['groups'] if g['split']=='train' and any(x['label']==1 for x in g['candidates'])]
    if not groups or not any(g['split']=='val' for g in m['groups']): raise ValueError('Train positives and separate validation required')
    heads=create_heads(c).to(device)
    saved=phase('read resume',lambda:torch.load(resume,map_location='cpu',weights_only=True) if resume else None)
    if saved is not None:
        if saved.get('run_identity')!=run_identity: raise ValueError('Resume requires identical DDP data/cache/config/code/devices')
        if Path(resume).resolve().parent!=out or not (out/'best.pth').exists(): raise ValueError('Resume in original DDP run directory retaining best.pth')
        heads.load_state_dict(saved['state_dict'],strict=True)
    ddp=wrap_ddp(MLPGroupLoss(heads,arrays,device,tc['temperature']),device)
    parameters=list(heads.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=tc['lr'],weight_decay=tc['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=tc['epochs'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)  # Preserve original FP32 MLP training.
    rng=random.Random(c['seed']); torch.manual_seed(c['seed']+100003*rank)
    if device.startswith('cuda:'): torch.cuda.manual_seed(c['seed']+100003*rank)
    history=[]; first=0; best=-1.; stale=0; hard={}; stopped=False
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        history=saved['history']; first=saved['epoch']+1; best=saved['best_score']; stale=saved['stale']; hard=saved['hard']
        state=saved['rng_by_rank'][rank]; rng.setstate(state['python']); torch.set_rng_state(state['torch'])
        if device.startswith('cuda:'): torch.cuda.set_rng_state(state['cuda'],device)
    phase('write run metadata',lambda:write_json(out/'run.json',{'run_identity':run_identity,'synthetic':m['synthetic'],
        'global_batch_groups':tc['batch_size'],'gradient_sync':'torch.nn.parallel.DistributedDataParallel / Gloo',
        'trainable_parameters':sum(p.numel() for p in parameters),'DINO_loaded_for_training':False,
        'validation_rank':0,'validation_all_candidates':True,'diagnostic_only':diagnostic,
        'training_tail':'zero-loss slots; exact mean over real groups'}) if rank==0 else None)
    for epoch in range(first,tc['epochs']):
        if stale>=tc['patience']: break
        write_json(out/f'rank_{rank}_status.json',{'stage':'mining','epoch':epoch,'device':device})
        if not hard or epoch%tc['hard_refresh_epochs']==0:
            part=phase('train-only hard mining',lambda:mine_hard_negatives(heads,groups[rank::world],arrays,device,tc['hard_negative_count']))
            parts=gather(part); hard={k:v for p in parts for k,v in p.items()}
            if len(hard)!=sum(len(p) for p in parts) or set(hard)!={g['id'] for g in groups}: raise ValueError('Mining shard mismatch')
            phase('save mined negatives',lambda:write_json(out/f'hard_negatives_epoch_{epoch:03}.json',
                {'split':'train','cache_fingerprint':index['fingerprint'],'selected_indices':hard}) if rank==0 else None)
        schedule=schedule_groups(groups,rng); ddp.train(); losses=[]; processed=[]; updates=0; gradients={'ir':0.,'satellite':0.}
        if device.startswith('cuda:'): torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        for step,start in enumerate(range(0,len(schedule),tc['batch_size'])):
            window=schedule[start:start+tc['batch_size']]
            local=shard_window(window,rank,world)
            items=[(g,hard[g['id']]) if g is not None else None for g in local]
            write_json(out/f'rank_{rank}_status.json',{'stage':'training','epoch':epoch,'step':step,'device':device,
                'ids':[g['id'] for g in local if g is not None]})
            optimizer.zero_grad(set_to_none=True)
            losses.extend(backward_window(ddp,items,len(window),scaler))
            if not all(gather(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters))):
                raise ValueError('Nonfinite or missing DDP MLP gradients')
            for domain in gradients: gradients[domain]+=sum(float(p.grad.norm()) for p in getattr(heads,domain).parameters())
            torch.nn.utils.clip_grad_norm_(parameters,5.); optimizer.step(); updates+=1
            processed.extend(g['id'] for g in local if g is not None)
        if updates==0: raise ValueError('No MLP optimizer updates')
        if device.startswith('cuda:'): torch.cuda.synchronize(device)
        stats=gather({'rank':rank,'device':device,'query_groups':len(losses),'loss_sum':sum(losses),
            'train_seconds':time.perf_counter()-started,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda:') else 0})
        scheduler.step(); fingerprints=gather(state_fingerprint(heads.state_dict()))
        if len(set(fingerprints))!=1: raise RuntimeError('DDP MLP replicas diverged')
        write_json(out/f'rank_{rank}_epoch_{epoch:03}.json',{'processed_ids':processed,'stats':stats[rank],'head_sha256':fingerprints[rank]})
        report=phase('rank0 full-candidate validation',lambda:evaluate_arrays(m,arrays,heads,'val',device) if rank==0 else None)
        score=gather(report['results']['heads']['overall']['recall_all_legal']['1'] if rank==0 else None)[0]
        if score is None: raise ValueError('Empty validation set')
        improved=score>best
        if improved: best=score; stale=0
        else: stale+=1
        count=sum(s['query_groups'] for s in stats)
        row={'epoch':epoch,'loss':sum(s['loss_sum'] for s in stats)/count,'val_recall_at_1_all':score,
            'gradient_norm_sum':gradients,'optimizer_updates':updates,'query_groups':count,'replica_head_hashes':fingerprints}
        history.append(row)
        states=gather({'python':rng.getstate(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.startswith('cuda:') else None})
        def checkpoint():
            if rank!=0: return
            metadata={'format_version':1,'state_dict':{k:v.detach().cpu().clone() for k,v in heads.state_dict().items()},
                'structure':heads.structure,'config':c,'epoch':epoch,'val_recall_at_1_all':score,'manifest_id':m['manifest_id'],
                'cache_fingerprint':index['fingerprint'],'backbone':index['backbone'],'synthetic':m['synthetic'],
                'selection_split':'val','run_identity':run_identity}
            if improved: atomic_save(out/'best.pth',metadata); write_json(out/'best_validation.json',report)
            atomic_save(out/'last.pth',{**metadata,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'history':history,'best_score':best,'stale':stale,'hard':hard,'rng_by_rank':states})
            write_training_history(out,history)
            write_json(out/f'performance_epoch_{epoch:03}.json',{'workers':stats,'global_query_groups_per_second':count/max(s['train_seconds'] for s in stats),
                'scope':'training forward/backward/communication and status I/O; excludes mining and validation'})
            print(row,flush=True)
        phase('rank0 checkpoint',checkpoint)
        if stop_after_epoch is not None and epoch>=stop_after_epoch:
            stopped=epoch+1<tc['epochs'] and stale<tc['patience']; break
    def finish():
        if rank!=0: return
        write_evaluation(out/'best_validation_review.json',read_json(out/'best_validation.json'),m,render_images=fixture is None)
        write_json(out/'training_result.json',{'status':'paused' if stopped else 'complete','epochs_recorded':len(history),
            'best':'best.pth','last':'last.pth','synthetic':m['synthetic'],'diagnostic_only':diagnostic})
    phase('final review output',finish)


def worker(rank,devices,uri,timeout,manifest,cache_dir,output,resume,diagnostic,stop_after_epoch,fixture):
    out=Path(output)
    with (out/f'rank_{rank}.log').open('a',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            init_group(rank,devices,uri,timeout)
            m=phase('verify manifest',lambda:read_manifest(manifest))
            _train(rank,devices,m,cache_dir,out,resume,diagnostic,stop_after_epoch,fixture)
            write_json(out/f'rank_{rank}_status.json',{'stage':'finished','device':devices[rank]})
        except BaseException as exc:
            write_json(out/f'rank_{rank}_failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
            traceback.print_exc(); raise
        finally:
            if dist.is_initialized(): dist.destroy_process_group()


def launch(manifest,cache_dir,output,devices=None,resume=None,diagnostic=False,stop_after_epoch=None,fixture=None):
    devices=list(devices or ['cuda:0','cuda:1']); validate_devices(devices,diagnostic)
    m=read_manifest(manifest); validate_settings(m['config'])
    if (diagnostic or fixture) and not m['synthetic']: raise ValueError('Diagnostics are forbidden on real data')
    if fixture and not diagnostic: raise ValueError('Fixture requires explicit diagnostic mode')
    if stop_after_epoch is not None and stop_after_epoch<0: raise ValueError('Invalid stop epoch')
    out=Path(output).resolve()
    if resume:
        if not out.is_dir() or not (out/'run.json').exists(): raise ValueError('Resume requires an existing DDP run directory')
    else: fresh_directory(out)
    write_json(out/'job_status.json',{'status':'running','devices':devices,'diagnostic_only':diagnostic,'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        spawn_workers(worker,devices,out,(str(Path(manifest).resolve()),str(Path(cache_dir).resolve()),str(out),
            str(Path(resume).resolve()) if resume else None,diagnostic,stop_after_epoch,fixture))
        write_json(out/'job_status.json',dict(read_json(out/'training_result.json'),devices=devices))
    except BaseException as exc:
        write_json(out/'job_status.json',{'status':'failed','error':str(exc),'rank_failure_reports':[p.name for p in out.glob('rank_*_failure.json')]})
        raise
    return out/'best.pth'


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--cache',required=True)
    p.add_argument('--output',default='artifacts/real/mlp_ddp'); p.add_argument('--devices',default='cuda:0,cuda:1'); p.add_argument('--resume')
    p.add_argument('--diagnostic',action='store_true',help='Synthetic-only CPU/shared-GPU process test')
    p.add_argument('--stop-after-epoch',type=int,help='Graceful stop at zero-based epoch for resume testing')
    a=p.parse_args(); print(launch(a.manifest,a.cache,a.output,a.devices.split(','),a.resume,a.diagnostic,a.stop_after_epoch))
