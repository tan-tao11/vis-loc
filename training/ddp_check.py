"""Small DDP communication/gradient check before loading a large backbone."""
import argparse
import contextlib
from pathlib import Path
import traceback
import torch
from torch import nn
import torch.distributed as dist
from common import read_json, write_json
from outputs import fresh_directory
from ddp_runtime import init_group, validate_devices, spawn_workers, wrap_ddp, gather


def check_worker(rank,devices,uri,timeout_seconds,output):
    out=Path(output)
    with (out/f'rank_{rank}.log').open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            device=init_group(rank,devices,uri,timeout_seconds)
            layer=nn.Linear(1,1,bias=False,device=device)
            with torch.no_grad(): layer.weight.zero_()
            ddp=wrap_ddp(layer,device); opt=torch.optim.SGD(ddp.parameters(),lr=.1)
            target=torch.tensor([[float(rank+1)]],device=device)
            loss=(ddp(torch.ones_like(target))-target).square().mean(); loss.backward()
            # Local gradients -2 and -4, DDP average -3, resulting weight +0.3.
            assert torch.allclose(layer.weight.grad,torch.tensor([[-3.]],device=device))
            opt.step(); assert torch.allclose(layer.weight,torch.tensor([[.3]],device=device))
            weights=gather(float(layer.weight.detach().cpu()))
            if len(set(weights))!=1: raise AssertionError('DDP replicas differ')
            record={'rank':rank,'device':device,'device_name':torch.cuda.get_device_name(device) if device.startswith('cuda:') else 'CPU',
                'gloo_all_reduce':'passed','ddp_mean_gradient':-3.,'weight_after_step':weights[rank],'replicas_equal':True}
            write_json(out/f'rank_{rank}.json',record)
        except BaseException as exc:
            write_json(out/f'rank_{rank}_failure.json',{'error':str(exc),'traceback':traceback.format_exc()}); raise
        finally:
            if dist.is_initialized(): dist.destroy_process_group()


def check(output,devices=('cuda:0','cuda:1'),diagnostic=False):
    validate_devices(list(devices),diagnostic)
    out=fresh_directory(output).resolve()
    write_json(out/'status.json',{'status':'running','devices':devices,'diagnostic_only':diagnostic})
    try:
        spawn_workers(check_worker,list(devices),out,(str(out),),60)
        result={'status':'passed','torch_version':str(torch.__version__),'backend':'gloo','world_size':2,
            'diagnostic_only':diagnostic,'distinct_gpu_test_passed':not diagnostic,
            'ranks':[read_json(out/f'rank_{r}.json') for r in range(2)]}
        write_json(out/'status.json',result); return result
    except BaseException as exc:
        write_json(out/'status.json',{'status':'failed','error':str(exc),'diagnostic_only':diagnostic}); raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--devices',default='cuda:0,cuda:1');p.add_argument('--output',default='artifacts/ddp_check')
    p.add_argument('--diagnostic',action='store_true',help='Permit synthetic CPU/shared-GPU tests; not a two-card acceptance result')
    a=p.parse_args(); print(check(a.output,a.devices.split(','),a.diagnostic))
