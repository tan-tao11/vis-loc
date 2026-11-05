import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from common import ROOT,config,read_json,digest
from appearance_augmentation import AppearanceAugmenter,AugmentedImages,TrainingViews,save_augmentation_preview,validate_augmentation_settings
from lora_data import OnlineImages
from lora import LoRARetriever
from synthetic import make_fixture
from prepare_dataset import prepare
from tests.test_lora import TinyDino


def profile(): return config(ROOT/'configs/lora_aug.yaml')['augmentation']


class TinyImageDino(TinyDino):
    def prepare_tokens_with_masks(self,x):
        return torch.nn.functional.adaptive_avg_pool2d(x,(32,32)).flatten(2)


def tiny_aug_factory(m,c,device):
    import torch.distributed as dist
    return LoRARetriever(TinyImageDino(),c['lora']).to(device),{'synthetic_test_backbone':True},\
        AugmentedImages(m,c['augmentation'],c['seed']+200003+100003*dist.get_rank())


def prepared(tmp_path):
    c=config(make_fixture(tmp_path/'data')); m=prepare(c)
    return m,Path(c['output'])/'manifest.json'


def small_settings(ddp=False):
    c=config(ROOT/('configs/lora_aug_ddp.yaml' if ddp else 'configs/lora_aug.yaml'))
    c['lora'].update(last_n_blocks=1,rank=2,alpha=4.,dropout=.1)
    c['train'].update(epochs=3,patience=4,amp=False,max_positives=1,hard_negatives=1,random_negatives=1)
    c['evaluation']['image_batch_size']=8
    if not ddp: c['train'].update(accumulation_groups=4,device='cpu')
    return c


@pytest.mark.parametrize('change',[{'enabled':'true'},{'preserve_probability':-1},{'gamma_range':[0,1]},
    {'noise_std_range':[.1,.2]},{'brightness_delta':float('nan')},{'rotation_degrees':10}])
def test_reject_invalid_or_geometric_options(change):
    c=profile(); c.update(change)
    with pytest.raises(ValueError): validate_augmentation_settings(c)


def test_grayscale_bounds_determinism_independent_rng_and_no_input_mutation():
    c=profile(); c.update(preserve_probability=0.,noise_probability=1.,blur_probability=1.)
    a=AppearanceAugmenter(c,3); b=AppearanceAugmenter(c,3); other=AppearanceAugmenter(c,4)
    original=np.arange(1024,dtype=np.uint8).reshape(32,32); before=original.copy()
    state=torch.get_rng_state().clone()
    x,info=a.apply(original,'ir'); y,_=b.apply(original,'ir'); z,_=other.apply(original,'ir')
    assert np.array_equal(x,y) and not np.array_equal(x,z) and not np.array_equal(x,original)
    assert np.array_equal(original,before) and torch.equal(state,torch.get_rng_state())
    assert x.dtype==np.uint8 and x.shape==original.shape and x.min()>=0 and x.max()<=255
    assert info['noise_std']>0 and info['blur_sigma']>0
    saved=a.state_dict(); expected,_=a.apply(original,'satellite')
    a.load_state_dict(saved); restored,_=a.apply(original,'satellite')
    assert np.array_equal(expected,restored)
    wrong=AppearanceAugmenter(dict(c,enabled=False),3)
    with pytest.raises(ValueError,match='settings mismatch'): wrong.load_state_dict(saved)


@pytest.mark.parametrize('settings',[{'enabled':False},{'preserve_probability':1.}])
def test_disable_or_preserve_returns_exact_original(settings):
    c=profile(); c.update(settings); augmenter=AppearanceAugmenter(c,8)
    source=np.arange(256,dtype=np.uint8).reshape(16,16)
    result,info=augmenter.apply(source,'satellite')
    assert np.array_equal(result,source) and not info['applied']
    assert augmenter.counts['satellite']['kept']==1


def test_clean_mining_eval_channels_and_preview_do_not_change_training_rng(tmp_path):
    from train_lora import mine_online
    from evaluate_lora import collect_online
    from preprocess import tensor_image
    m,_=prepared(tmp_path); original_manifest=digest(m)
    c=small_settings(); c['augmentation']['preserve_probability']=0.
    torch.set_num_threads(2); model=LoRARetriever(TinyImageDino(),c['lora'])
    with OnlineImages(m) as baseline,AugmentedImages(m,c['augmentation'],123) as images:
        state=images.augmentation.state_dict()
        for g in m['groups']:
            assert torch.equal(images.read(g),baseline.read(g))
            assert torch.equal(images.read(g,0),baseline.read(g,0))
            if g['split']!='train':
                with pytest.raises(ValueError,match='restricted'): TrainingViews(images).read(g)
        train_groups=[g for g in m['groups'] if g['split']=='train']
        mine_online(model,images,train_groups,c['train'],'cpu')
        collect_online(model,images,m,'val','cpu',8,False)
        save_augmentation_preview(images,train_groups,tmp_path,12)
        assert torch.equal(state['rng_state'],images.augmentation.state_dict()['rng_state'])
        assert state['counts']==images.augmentation.counts
        image=TrainingViews(images).read(train_groups[0])
        mean=torch.tensor(m['config']['preprocess']['mean'])[:,None,None]
        std=torch.tensor(m['config']['preprocess']['std'])[:,None,None]
        raw=image*std+mean
        torch.testing.assert_close(raw[0],raw[1],rtol=0,atol=2e-7)
        torch.testing.assert_close(raw[1],raw[2],rtol=0,atol=2e-7)
    assert digest(m)==original_manifest
    assert (tmp_path/'augmentation_preview.png').exists()
    assert read_json(tmp_path/'augmentation_preview.json')['training_rng_unchanged']


def test_single_aug_resume_exact_and_disabled_equals_original_trainer(tmp_path,monkeypatch):
    import train_lora_aug as trainer
    import train_lora as original
    m,manifest=prepared(tmp_path); c=small_settings(); torch.set_num_threads(2)
    factory=lambda _,spec,device:(LoRARetriever(TinyImageDino(),spec).to(device),{'fixture':True})
    monkeypatch.setattr(trainer,'build_lora',factory); monkeypatch.setattr(original,'build_lora',factory)
    trainer.train_lora(c,manifest,tmp_path/'full','cpu')
    trainer.train_lora(c,manifest,tmp_path/'resumed','cpu',stop_after_epoch=0)
    trainer.train_lora(c,manifest,tmp_path/'resumed','cpu',tmp_path/'resumed/last.pth')
    full=torch.load(tmp_path/'full/last.pth',weights_only=True); resumed=torch.load(tmp_path/'resumed/last.pth',weights_only=True)
    # Peak GPU memory is zero on CPU; history includes cumulative augmentation counts.
    assert full['history']==resumed['history']
    assert full['scheduler']==resumed['scheduler']
    assert torch.equal(full['rng_augmentation']['rng_state'],resumed['rng_augmentation']['rng_state'])
    assert all(torch.equal(value,resumed['state_dict'][key]) for key,value in full['state_dict'].items())
    eligible=sum(g['split']=='train' and any(x['label']==1 for x in g['candidates']) and any(x['label']==0 for x in g['candidates']) for g in m['groups'])
    assert full['rng_augmentation']['counts']['ir']['seen']==eligible*3
    assert full['rng_augmentation']['counts']['satellite']['seen']==eligible*3*3
    c['augmentation']['enabled']=False
    trainer.train_lora(c,manifest,tmp_path/'disabled','cpu')
    original.train_lora(c,manifest,tmp_path/'baseline','cpu')
    disabled=torch.load(tmp_path/'disabled/last.pth',weights_only=True); base=torch.load(tmp_path/'baseline/last.pth',weights_only=True)
    assert all(torch.equal(value,disabled['state_dict'][key]) for key,value in base['state_dict'].items())


def test_two_process_aug_resume_restores_rank_rng_and_same_replicas(tmp_path):
    from train_lora_aug_ddp import launch
    m,manifest=prepared(tmp_path); c=small_settings(True)
    launch(c,manifest,tmp_path/'full',['cpu','cpu'],diagnostic=True,fixture=tiny_aug_factory)
    launch(c,manifest,tmp_path/'resumed',['cpu','cpu'],diagnostic=True,stop_after_epoch=0,fixture=tiny_aug_factory)
    launch(c,manifest,tmp_path/'resumed',['cpu','cpu'],resume=tmp_path/'resumed/last.pth',diagnostic=True,fixture=tiny_aug_factory)
    full=torch.load(tmp_path/'full/last.pth',weights_only=True); resumed=torch.load(tmp_path/'resumed/last.pth',weights_only=True)
    assert full['history']==resumed['history'] and full['scheduler']==resumed['scheduler']
    assert all(torch.equal(value,resumed['state_dict'][key]) for key,value in full['state_dict'].items())
    for row in full['history']: assert len(set(row['replica_adapter_hashes']))==1
    assert not torch.equal(full['rng_by_rank'][0]['augmentation']['rng_state'],full['rng_by_rank'][1]['augmentation']['rng_state'])
    eligible=sum(g['split']=='train' and any(x['label']==1 for x in g['candidates']) and any(x['label']==0 for x in g['candidates']) for g in m['groups'])
    assert eligible%2==0
    for rank in range(2):
        a=full['rng_by_rank'][rank]['augmentation']; b=resumed['rng_by_rank'][rank]['augmentation']
        assert torch.equal(a['rng_state'],b['rng_state']) and a['counts']==b['counts']
        assert a['counts']['ir']['seen']==eligible//2*3 and a['counts']['satellite']['seen']==eligible//2*3*3
    assert read_json(tmp_path/'resumed/job_status.json')['status']=='complete'
