"""Optional two-process Gloo DDP LoRA trainer. Existing single-GPU entry stays unchanged."""
import argparse
import contextlib
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import torch.distributed as dist
from common import ROOT, config, seed_all, digest, sha256, read_json, write_json
from outputs import fresh_directory, write_training_history, write_evaluation
from lora import build_lora
from lora_data import load_manifest, OnlineImages, balanced_schedule, eligible_training_groups, sample_candidates
from train_lora import validate_settings as validate_single_settings, mine_online, atomic_save, state_fingerprint
from evaluate_lora import collect_online
from evaluate import evaluate_arrays
from ddp_runtime import (gather, phase, init_group, validate_devices, spawn_workers, shard_window,
                         GroupLoss, wrap_ddp, backward_window, merge_evaluation)


def validate_settings(c, world_size):
    tc=c['train']; size=tc['global_batch_groups']
    if not isinstance(size,int) or isinstance(size,bool) or size<world_size or size%world_size:
        raise ValueError('global_batch_groups must be a positive multiple of world_size')
    validate_single_settings(dict(c,train=dict(tc,accumulation_groups=size)))
    dc=c['distributed']
    if dc['backend']!='gloo': raise ValueError('Windows optional trainer implements Gloo only')
    if not isinstance(dc['timeout_seconds'],int) or dc['timeout_seconds']<10: raise ValueError('timeout_seconds must be >=10')
    if not isinstance(dc['cpu_threads_per_worker'],int) or dc['cpu_threads_per_worker']<1: raise ValueError('Invalid worker CPU thread count')


def identity(c,m,backbone,devices,diagnostic,fixture):
    files=('train_lora_ddp.py','ddp_runtime.py','train_lora.py','lora.py','lora_data.py','losses.py','evaluate.py','evaluate_lora.py','models.py')
    return {'method':'lora_ddp_v1','manifest_id':m['manifest_id'],'settings':digest(c),'backbone':backbone,
        'world_size':len(devices),'devices':devices,'diagnostic_only':diagnostic,
        'fixture_factory':None if fixture is None else fixture.__module__+'.'+fixture.__qualname__,
        'torch_version':str(torch.__version__),'code':{f:sha256(ROOT/f) for f in files}}


def distributed_evaluation(model,images,m,split,device,batch,amp):
    selected=[g for g in m['groups'] if g['split']==split][dist.get_rank()::dist.get_world_size()]
    def evaluate_part():
        if not selected: return {'rows':[],'seconds':0.}
        local=dict(m,groups=selected)
        arrays,seconds=collect_online(model,images,local,split,device,batch,amp)
        rows=evaluate_arrays(local,arrays,split=split)['results']['gem1024']['rows']
        return {'rows':rows,'seconds':seconds}
    part=phase('full-candidate '+split,evaluate_part)
    return merge_evaluation(m,gather(part),split)


def _train(rank,devices,c,m,out,resume,diagnostic,stop_after_epoch,fixture):
    device=devices[rank]; world=len(devices); tc=c['train']
    torch.set_num_threads(c['distributed']['cpu_threads_per_worker'])
    torch.backends.cuda.matmul.allow_tf32=bool(tc['allow_tf32']); torch.backends.cudnn.allow_tf32=bool(tc['allow_tf32'])
    seed_all(c['seed'])
    built=phase('load local frozen DINO',lambda: fixture(m,c,device) if fixture else (*build_lora(m['config'],c['lora'],device),OnlineImages(m)))
    model,backbone,images=built
    run_identity=identity(c,m,backbone,devices,diagnostic,fixture)
    identities=gather(digest(run_identity))
    if len(set(identities))!=1: raise ValueError('Rank data/config/backbone identities differ')
    groups=eligible_training_groups(m)
    if not groups or not any(g['split']=='val' for g in m['groups']): raise ValueError('Train positives/negatives and separate validation required')
    saved=phase('read resume',lambda: torch.load(resume,map_location='cpu',weights_only=True) if resume else None)
    if saved is not None:
        if saved.get('run_identity')!=run_identity: raise ValueError('DDP resume requires identical data/config/code/world size/devices; single-GPU checkpoints cannot resume here')
        if Path(resume).resolve().parent!=out or not (out/'best.pth').exists(): raise ValueError('Resume in original DDP run directory retaining best.pth')
        model.load_adapter_state(saved['state_dict'])
    rng=random.Random(c['seed']+100003*rank)
    # Initialization is shared; dropout/sample RNG streams are rank-specific.
    torch.manual_seed(c['seed']+100003*rank)
    if device.startswith('cuda:'): torch.cuda.manual_seed(c['seed']+100003*rank)
    with images:
        wrapped=GroupLoss(model,images,device,tc)
        ddp=wrap_ddp(wrapped,device)
        parameters=list(wrapped.adapter_parameters)
        optimizer=torch.optim.AdamW(parameters,lr=tc['lr'],weight_decay=tc['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=tc['epochs'])
        scaler=torch.amp.GradScaler('cuda',enabled=tc['amp'] and device.startswith('cuda:'),init_scale=1024.)
        history=[]; first=0; best_key=None; stale=0; hard={}
        if saved is not None:
            optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
            history=saved['history']; first=saved['epoch']+1; best_key=saved['best_key']; stale=saved['stale']; hard=saved['hard']
            state=saved['rng_by_rank'][rank]; rng.setstate(state['python']); torch.set_rng_state(state['torch'])
            if device.startswith('cuda:'): torch.cuda.set_rng_state(state['cuda'],device)
        def save_run():
            if rank==0: write_json(out/'run.json',{'run_identity':run_identity,'settings':c,'synthetic':m['synthetic'],
                'diagnostic_only':diagnostic,'distinct_gpu_training':not diagnostic,'trainable_parameters':model.assert_freeze(),
                'global_batch_groups':tc['global_batch_groups'],'local_groups_per_full_step':tc['global_batch_groups']//world,
                'base_frozen_and_identical_by_hash':True,'gradient_sync':'torch.nn.parallel.DistributedDataParallel / Gloo',
                'validation_query_padding':False,'training_tail':'zero-loss slots; weighted mean over real groups'})
        phase('write run metadata',save_run)
        stopped=False
        for epoch in range(first,tc['epochs']):
            if stale>=tc['patience']: break
            write_json(out/f'rank_{rank}_status.json',{'stage':'mining','epoch':epoch,'device':device})
            if not hard or epoch%tc['hard_refresh_epochs']==0:
                mine=phase('train-only hard mining',lambda: mine_online(model,images,groups[rank::world],tc,device))
                parts=gather(mine); hard={k:v for part in parts for k,v in part.items()}
                if len(hard)!=sum(len(p) for p in parts) or set(hard)!={g['id'] for g in groups}: raise ValueError('Hard mining shard mismatch')
                phase('save mined negatives',lambda: write_json(out/f'hard_negatives_epoch_{epoch:03}.json',
                    {'split':'train','manifest_id':m['manifest_id'],'indices':hard,'world_size':world}) if rank==0 else None)
            schedule=balanced_schedule(groups,random.Random(c['seed']+epoch))
            ddp.train(); losses=[]; updates=0; overflows=0; processed=[]; grads={'A':0.,'B':0.}
            if device.startswith('cuda:'): torch.cuda.reset_peak_memory_stats(device)
            started=time.perf_counter()
            for step,start in enumerate(range(0,len(schedule),tc['global_batch_groups'])):
                window=schedule[start:start+tc['global_batch_groups']]
                local=shard_window(window,rank,world)
                items=phase('sample candidates',lambda: [(g,sample_candidates(g,hard[g['id']],tc,rng)) if g else None for g in local])
                write_json(out/f'rank_{rank}_status.json',{'stage':'training','epoch':epoch,'step':step,'device':device,
                    'ids':[g['id'] for g in local if g is not None]})
                optimizer.zero_grad(set_to_none=True)
                losses.extend(backward_window(ddp,items,len(window),scaler))
                scaler.unscale_(optimizer); model.assert_freeze()
                finite=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
                flags=gather(finite)
                if len(set(flags))!=1: raise RuntimeError('Ranks disagree on gradient validity; no optimizer step accepted')
                if finite:
                    for name,p in model.named_parameters():
                        if p.requires_grad: grads['A' if '.lora_A.' in name else 'B']+=float(p.grad.norm())
                    torch.nn.utils.clip_grad_norm_(parameters,tc['clip_grad_norm']); updates+=1
                else:
                    if not scaler.is_enabled(): raise ValueError('Nonfinite DDP gradients')
                    overflows+=1
                scaler.step(optimizer); scaler.update()
                processed.extend(g['id'] for g in local if g is not None)
            if updates==0: raise RuntimeError('No successful DDP optimizer update')
            if device.startswith('cuda:'): torch.cuda.synchronize(device)
            stats=gather({'rank':rank,'device':device,'query_groups':len(losses),'loss_sum':sum(losses),
                'train_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda:') else 0})
            scheduler.step(); model.assert_freeze()
            fingerprints=gather(state_fingerprint(model.adapter_state()))
            if len(set(fingerprints))!=1: raise RuntimeError('DDP adapter replicas diverged')
            write_json(out/f'rank_{rank}_epoch_{epoch:03}.json',{'processed_ids':processed,'stats':stats[rank],'adapter_sha256':fingerprints[rank]})
            report=distributed_evaluation(model,images,m,'val',device,c['evaluation']['image_batch_size'],tc['amp'])
            metrics=report['results']['lora_gem1024']['overall']; recall=metrics['recall_all_legal']; rank_mean=metrics['mean_first_positive_rank_covered']
            key=[recall['1'],recall['5'],-rank_mean if rank_mean is not None else -1e30]
            if key[0] is None: raise ValueError('Empty validation')
            improved=best_key is None or key[0]>best_key[0]+tc['min_delta'] or (key[0]==best_key[0] and key[1:]>best_key[1:])
            if improved: best_key=key; stale=0
            else: stale+=1
            count=sum(s['query_groups'] for s in stats)
            row={'epoch':epoch,'loss':sum(s['loss_sum'] for s in stats)/count,'val_recall_at_1_all':recall['1'],
                'val_recall_all':recall,'optimizer_updates':updates,'amp_overflow_updates_skipped':overflows,
                'gradient_norm_sum':grads,'query_groups':count,'replica_adapter_hashes':fingerprints}
            history.append(row)
            rng_states=gather({'python':rng.getstate(),'torch':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state(device) if device.startswith('cuda:') else None})
            metadata={'kind':'scc_dinov2_lora_training','format_version':1,'lora':c['lora'],'state_dict':model.adapter_state(),
                'backbone':backbone,'preprocess':m['config']['preprocess'],'pose':m['config']['pose'],
                'gem':{'p':4.,'clamp_min':1e-6,'dimension':1024},'output_dim':1024,'manifest_id':m['manifest_id'],
                'synthetic_only':m['synthetic'],'selection_split':'val','epoch':epoch,'val_selection_key':key,
                'run_identity':run_identity,'distributed':{'world_size':world,'backend':'gloo','diagnostic_only':diagnostic}}
            def checkpoint():
                if rank!=0: return
                if improved: atomic_save(out/'best.pth',metadata); write_json(out/'best_validation.json',report)
                atomic_save(out/'last.pth',{**metadata,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'scaler':scaler.state_dict(),'history':history,'best_key':best_key,'stale':stale,'hard':hard,'rng_by_rank':rng_states})
                write_training_history(out,history)
                write_json(out/f'performance_epoch_{epoch:03}.json',{'workers':stats,'global_query_groups_per_second':count/max(s['train_seconds'] for s in stats),
                    'scope':'training forward/backward/communication and rank-status I/O; excludes mining and validation'})
                print(row,flush=True)
            phase('rank0 checkpoint',checkpoint)
            if stop_after_epoch is not None and epoch>=stop_after_epoch:
                stopped=epoch+1<tc['epochs'] and stale<tc['patience']; break
            if stale>=tc['patience']: break
        def final_report():
            if rank==0:
                write_evaluation(out/'best_validation_review.json',read_json(out/'best_validation.json'),m,render_images=fixture is None)
                write_json(out/'training_result.json',{'status':'paused' if stopped else 'complete','epochs_recorded':len(history),
                    'best':'best.pth','last':'last.pth','synthetic':m['synthetic'],'diagnostic_only':diagnostic})
        phase('final report',final_report)


def worker(rank,devices,uri,timeout_seconds,c,manifest_file,output,resume,diagnostic,stop_after_epoch,fixture):
    out=Path(output)
    with (out/f'rank_{rank}.log').open('a',encoding='utf-8',buffering=1) as log, contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            init_group(rank,devices,uri,timeout_seconds)
            m=phase('load manifest',lambda:load_manifest(manifest_file))
            _train(rank,devices,c,m,out,resume,diagnostic,stop_after_epoch,fixture)
            write_json(out/f'rank_{rank}_status.json',{'stage':'finished','device':devices[rank]})
        except BaseException as exc:
            write_json(out/f'rank_{rank}_failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
            traceback.print_exc(); raise
        finally:
            if dist.is_initialized(): dist.destroy_process_group()


def launch(c,manifest_file,output,devices=None,resume=None,diagnostic=False,stop_after_epoch=None,fixture=None):
    devices=list(devices or c['distributed']['devices'])
    validate_settings(c,len(devices)); validate_devices(devices,diagnostic)
    m=load_manifest(manifest_file)
    if (diagnostic or fixture) and not m['synthetic']: raise ValueError('Shared-GPU/CPU fixture diagnostics are forbidden on real data')
    if fixture is not None and not diagnostic: raise ValueError('Fixture requires explicit diagnostic mode')
    if stop_after_epoch is not None and stop_after_epoch<0: raise ValueError('Invalid stop epoch')
    out=Path(output).resolve()
    if resume:
        if not out.is_dir() or not (out/'run.json').exists(): raise ValueError('Resume requires an existing DDP output directory')
    else: fresh_directory(out)
    write_json(out/'job_status.json',{'status':'running','devices':devices,'diagnostic_only':diagnostic,'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        spawn_workers(worker,devices,out,(c,str(Path(manifest_file).resolve()),str(out),str(Path(resume).resolve()) if resume else None,
            diagnostic,stop_after_epoch,fixture),c['distributed']['timeout_seconds'])
        result=read_json(out/'training_result.json')
        write_json(out/'job_status.json',dict(result,devices=devices))
    except BaseException as exc:
        write_json(out/'job_status.json',{'status':'failed','error':str(exc),'rank_failure_reports':[p.name for p in out.glob('rank_*_failure.json')]})
        raise
    return out/'best.pth'


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/lora_ddp.yaml'); p.add_argument('--manifest',required=True)
    p.add_argument('--output',default='artifacts/real/lora_ddp'); p.add_argument('--devices',help='Example: cuda:0,cuda:1')
    p.add_argument('--resume'); p.add_argument('--diagnostic',action='store_true',help='Explicit synthetic-only CPU/shared-GPU process test')
    p.add_argument('--stop-after-epoch',type=int,help='Graceful epoch-boundary stop for resume testing; zero-based')
    a=p.parse_args()
    print(launch(config(a.config),a.manifest,a.output,a.devices.split(',') if a.devices else None,a.resume,a.diagnostic,a.stop_after_epoch))
