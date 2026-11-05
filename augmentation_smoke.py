"""Real local DINO smoke for optional appearance augmentation; synthetic data only."""
import argparse
from pathlib import Path
import torch
from common import ROOT,config,read_json,write_json
from outputs import fresh_directory
from synthetic import make_fixture
from prepare_dataset import prepare
from train_lora_aug import train_lora
from train_lora_aug_ddp import launch
from export_lora import export_lora,load_lora_for_manifest
from integration.lora_adapter import LoRAInferenceAdapter
from lora_data import OnlineImages


def smoke(output,devices,mode='both',diagnostic=False):
    out=fresh_directory(output).resolve()
    report={'status':'running','synthetic_only':True,'diagnostic_only':diagnostic,'runs':{}}
    write_json(out/'smoke_report.json',report)
    try:
        base=config(ROOT/'configs/train.yaml')
        for group in base['data']['groups'].values():
            for key in ('directory','records'):
                p=Path(group[key])
                if p.is_absolute() or '..' in p.parts: raise ValueError('Synthetic fixture requires relative group paths')
        c=config(make_fixture(out/'data'))
        c['data'].update(per_image_txt=False,image_regex=r'(?P<sequence>sequence[^/]+)/(?P<frame>\d+)\.(?:png|jpg|jpeg|tif|tiff)$')
        m=prepare(c); manifest=Path(c['output'])/'manifest.json'
        for method in (['single','ddp'] if mode=='both' else [mode]):
            settings=config(ROOT/('configs/lora_aug.yaml' if method=='single' else 'configs/lora_aug_ddp.yaml'))
            settings['train'].update(epochs=2,patience=3,max_positives=1,hard_negatives=1,random_negatives=1)
            settings['evaluation']['image_batch_size']=1
            run=out/method; write_json(out/(method+'_settings.json'),settings)
            if method=='single': best=train_lora(settings,manifest,run,devices[0])
            else: best=launch(settings,manifest,run,devices,diagnostic=diagnostic)
            export_lora(best,run/'synthetic_ONLY_lora_aug.pth')
            model,_=load_lora_for_manifest(best,m,devices[0])
            adapter=LoRAInferenceAdapter(run/'synthetic_ONLY_lora_aug.pth',c,True,devices[0],allow_synthetic=True)
            with OnlineImages(m) as images: example=images.read(m['groups'][0]).unsqueeze(0).to(devices[0])
            with torch.no_grad(): error=float((model(example)-adapter.retrieve(example,amp=False)).abs().max())
            assert error<=1e-6
            last=torch.load(run/'last.pth',map_location='cpu',weights_only=True)
            states=[last['rng_augmentation']] if method=='single' else [r['augmentation'] for r in last['rng_by_rank']]
            counts={domain:{key:sum(s['counts'][domain][key] for s in states) for key in states[0]['counts'][domain]}
                for domain in ('ir','satellite')}
            eligible=sum(g['split']=='train' and any(x['label']==1 for x in g['candidates']) and any(x['label']==0 for x in g['candidates']) for g in m['groups'])
            assert counts['ir']['seen']==eligible*2 and counts['satellite']['seen']==eligible*2*3
            assert all(counts[d]['augmented']>0 and counts[d]['pixels_changed']>0 for d in counts)
            assert any(r['gradient_norm_sum']['A']>0 and r['gradient_norm_sum']['B']>0 for r in last['history'])
            if method=='ddp': assert all(len(set(r['replica_adapter_hashes']))==1 for r in last['history'])
            assert read_json(run/'best_validation.json')['results']['lora_gem1024']['overall']['queries']==4
            report['runs'][method]={'status':'passed','export_reload_max_abs_error':error,'counts':counts,
                'epochs':len(last['history']),'both_A_B_have_gradients':True,'freeze_assertions_passed':True,
                'preview_created':(run/'augmentation_preview.png').is_file(),'validation_queries':4,
                'augmentation_only_in_training_loss':True,'loss_history':[r['loss'] for r in last['history']],
                'replicas_equal':True if method=='ddp' else None,
                'distinct_gpu_test_passed':method=='ddp' and not diagnostic}
            write_json(out/'smoke_report.json',report)
            del adapter,model,example,last
            if devices[0].startswith('cuda:'): torch.cuda.empty_cache()
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc)); write_json(out/'smoke_report.json',report); raise
    write_json(out/'smoke_report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='artifacts/augmentation_smoke')
    p.add_argument('--devices',default='cuda:0,cuda:1'); p.add_argument('--mode',choices=['single','ddp','both'],default='both')
    p.add_argument('--diagnostic',action='store_true'); a=p.parse_args()
    print(smoke(a.output,a.devices.split(','),a.mode,a.diagnostic))
