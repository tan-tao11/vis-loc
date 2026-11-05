"""Shared mechanics for the optional Windows Gloo DDP path. No single-GPU edits."""
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
import math
from pathlib import Path
import traceback
import uuid
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from common import write_json
from evaluate import rank_metrics, summarize
from lora_data import encode_group
from losses import grouped_loss


def gather(value):
    values=[None]*dist.get_world_size(); dist.all_gather_object(values,value); return values


def phase(name, action):
    """Abort all ranks together for errors outside backward collectives."""
    result=None; error=None
    try: result=action()
    except Exception: error=f'rank {dist.get_rank()} / {name}\n'+traceback.format_exc()
    errors=gather(error)
    if any(errors): raise RuntimeError('\n'.join(x for x in errors if x))
    return result


def init_group(rank, devices, uri, timeout_seconds):
    device=devices[rank]
    if device.startswith('cuda:'): torch.cuda.set_device(torch.device(device))
    dist.init_process_group('gloo',init_method=uri,rank=rank,world_size=len(devices),timeout=timedelta(seconds=timeout_seconds))
    # Verify CUDA all-reduce on the actual selected devices, before loading DINO.
    probe=torch.tensor([rank+1.],device=device)
    dist.all_reduce(probe)
    if probe.item()!=len(devices)*(len(devices)+1)/2: raise RuntimeError('Gloo all-reduce produced an incorrect result')
    return device


def validate_devices(devices, diagnostic=False):
    if len(devices)!=2: raise ValueError('This optional launcher requires exactly two workers')
    if not diagnostic and (len(set(devices))!=2 or any(not x.startswith('cuda:') for x in devices)):
        raise ValueError('Production DDP requires two DISTINCT CUDA devices; diagnostics are explicitly opt-in')
    for value in devices:
        device=torch.device(value)
        if device.type not in ('cuda','cpu'): raise ValueError('Unsupported device')
        if device.type=='cuda' and (device.index is None or device.index>=torch.cuda.device_count()):
            raise ValueError(f'Requested {value}; only {torch.cuda.device_count()} visible CUDA device(s)')
    if not dist.is_available() or not dist.is_gloo_available(): raise RuntimeError('This PyTorch installation lacks Gloo')


def spawn_workers(worker, devices, output, args=(), timeout_seconds=1200):
    """FileStore avoids torchrun/libuv build differences. Spawn remains standard DDP."""
    out=Path(output).resolve()
    uri=(out/('rendezvous_'+uuid.uuid4().hex)).as_uri()  # Must be nonexistent for every launch, including resume.
    context=mp.spawn(worker,args=(devices,uri,timeout_seconds,*args),nprocs=len(devices),join=False)
    # ProcessContext detects a failed child and terminates peers; never silently keep half a run alive.
    while not context.join(timeout=1): pass


def shard_window(window, rank, world_size):
    """Zero-loss slots align backward counts without duplicating training examples."""
    slots=math.ceil(len(window)/world_size)
    return [window[i*world_size+rank] if i*world_size+rank<len(window) else None for i in range(slots)]


class GroupLoss(nn.Module):
    """One DDP forward owns the whole query/candidate graph, including microbatches.

    DDP registers only adapter parameters. Every worker independently verifies and
    loads the same frozen backbone; no 1.2 GB frozen-weight broadcast is needed.
    ParameterList holds the EXACT parameters used by the retriever, not copies.
    """
    def __init__(self, model, images, device, settings):
        super().__init__()
        self.adapter_parameters=nn.ParameterList([p for p in model.parameters() if p.requires_grad])
        self.runtime=[model,images]  # Intentionally not registered as a second module hierarchy.
        self.device=device; self.settings=settings

    def train(self, mode=True):
        super().train(mode)
        self.runtime[0].train(mode)
        return self

    def forward(self, item):
        if item is None:
            return sum(p.reshape(-1)[0]*0 for p in self.adapter_parameters)
        group,selected=item; tc=self.settings
        q,gallery=encode_group(self.runtime[0],self.runtime[1],group,self.device,tc['image_batch_size'],tc['amp'],selected)
        labels=torch.tensor([group['candidates'][i]['label'] for i in selected],device=self.device)
        loss,valid=grouped_loss(q,gallery,labels,tc['temperature'])
        if valid!=1 or not torch.isfinite(loss): raise ValueError('Invalid grouped training loss: '+group['id'])
        return loss


def wrap_ddp(module, device):
    return DDP(module,device_ids=[torch.device(device).index] if device.startswith('cuda:') else None,
               broadcast_buffers=False,find_unused_parameters=False)


def backward_window(ddp, items, global_count, scaler):
    if global_count<1 or not items: raise ValueError('Empty optimizer window')
    losses=[]
    for slot,item in enumerate(items):
        with ddp.no_sync() if slot<len(items)-1 else nullcontext():
            loss=phase('training forward',lambda: ddp(item))
            # DDP averages across ranks: this produces the exact mean over REAL
            # groups, even for a final window with unequal local counts.
            scaler.scale(loss*(dist.get_world_size()/global_count)).backward()
            if item is not None: losses.append(float(loss.detach()))
    return losses


def merge_evaluation(manifest, parts, split):
    groups={g['id']:g for g in manifest['groups'] if g['split']==split}
    rows=[row for part in parts for row in part['rows']]
    if not groups or len(rows)!=len(groups) or {r['id'] for r in rows}!=set(groups):
        raise ValueError('Distributed evaluation missing or duplicating queries')
    positions={key:i for i,key in enumerate(groups)}
    rows.sort(key=lambda r:positions[r['id']])
    buckets=defaultdict(list)
    for row in rows:
        g=groups[row['id']]
        for key in ('group','sequence','protocol'): buckets[key+':'+g[key]].append(row)
        if g['candidates']: buckets['block_m:%d'%(100*int(g['candidates'][0]['scale_m']//100))].append(row)
    result={'overall':summarize(rows),'rows':rows,'groups':{k:summarize(v) for k,v in buckets.items()},
        'online_feature_seconds':max(p['seconds'] for p in parts),
        'timing_scope':'maximum worker image I/O + DINO + ranking seconds; excludes loading and result gathering'}
    return {'split':split,'synthetic':manifest['synthetic'],'manifest_id':manifest['manifest_id'],
        'data_flow':{'matched':manifest['total_matched'],'processed':manifest['processed'],'legal':len(manifest['groups']),
            'preparation_failures':manifest['failures'],'limited':manifest['limited']},
        'evaluation_protocol':{'all_candidates':True,'candidate_subsampling':False,'world_size':len(parts),
            'query_padding':False,'exact_query_id_union_verified':True},'results':{'lora_gem1024':result}}
