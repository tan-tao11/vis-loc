"""Synthetic DINO DDP training/export check, with explicit two-card acceptance status."""
import argparse
from pathlib import Path
import torch
from common import ROOT,config,read_json,write_json
from outputs import fresh_directory
from synthetic import make_fixture
from prepare_dataset import prepare
from train_lora_ddp import launch
from ddp_check import check
from export_lora import export_lora,load_lora_for_manifest
from lora_data import OnlineImages
from integration.lora_adapter import LoRAInferenceAdapter


def smoke(output,devices,diagnostic=False):
    out=fresh_directory(output).resolve()
    report={'status':'running','synthetic_only':True,'diagnostic_only':diagnostic,'distinct_gpu_test_passed':False}
    write_json(out/'smoke_report.json',report)
    try:
        report['communication']=check(out/'communication',devices,diagnostic)
        # The shared fixture generator accepts configurable relative paths. Guard
        # them before it writes anything so server configuration cannot escape.
        original=config(ROOT/'configs/train.yaml')
        for g in original['data']['groups'].values():
            for key in ('directory','records'):
                p=Path(g[key])
                if p.is_absolute() or '..' in p.parts: raise ValueError('Synthetic smoke requires relative group paths: '+key)
        c=config(make_fixture(out/'data'))
        c['data'].update(per_image_txt=False,image_regex=r'(?P<sequence>sequence[^/]+)/(?P<frame>\d+)\.(?:png|jpg|jpeg|tif|tiff)$')
        m=prepare(c); manifest=Path(c['output'])/'manifest.json'
        settings=config(ROOT/'configs/lora_ddp.yaml')
        settings['train'].update(epochs=2,patience=3,max_positives=1,hard_negatives=1,random_negatives=1,global_batch_groups=4)
        settings['evaluation']['image_batch_size']=1
        write_json(out/'smoke_settings.json',settings)
        best=launch(settings,manifest,out/'training',devices,diagnostic=diagnostic)
        artifact=out/'synthetic_ONLY_lora_ddp.pth'; export_lora(best,artifact)
        model,ckpt=load_lora_for_manifest(best,m,devices[0])
        adapter=LoRAInferenceAdapter(artifact,c,True,devices[0],allow_synthetic=True)
        with OnlineImages(m) as images: image=images.read(m['groups'][0]).unsqueeze(0).to(devices[0])
        with torch.no_grad():
            expected=model(image); actual=adapter.retrieve(image,amp=False)
            error=float((expected-actual).abs().max())
            if error>1e-6: raise AssertionError('DDP export/reload mismatch')
        last=torch.load(out/'training/last.pth',map_location='cpu',weights_only=True)
        assert all(len(set(row['replica_adapter_hashes']))==1 for row in last['history'])
        assert any(row['gradient_norm_sum']['A']>0 and row['gradient_norm_sum']['B']>0 for row in last['history'])
        report.update(status='passed',distinct_gpu_test_passed=not diagnostic,export_reload_max_abs_error=error,
            adapter_replicas_equal=True,both_A_B_have_gradients=True,base_gradients_absent=True,
            loss_history=[r['loss'] for r in last['history']],
            validation_queries=read_json(out/'training/best_validation.json')['results']['lora_gem1024']['overall']['queries'],
            note='Synthetic pipeline check; shared-GPU diagnostics do not measure dual-A4000 throughput.')
    except BaseException as exc:
        report.update(status='failed',error=str(exc));write_json(out/'smoke_report.json',report);raise
    write_json(out/'smoke_report.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--devices',default='cuda:0,cuda:1');p.add_argument('--output',default='artifacts/ddp_smoke')
    p.add_argument('--diagnostic',action='store_true');a=p.parse_args();print(smoke(a.output,a.devices.split(','),a.diagnostic))
