"""Train-only grayscale appearance changes; no geometry or label mutations."""
import copy
import math
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from common import digest, write_json
from lora_data import OnlineImages
from preprocess import query_image, tensor_image


def validate_augmentation_settings(settings):
    expected={'enabled','preserve_probability','brightness_delta','contrast_range','gamma_range',
              'noise_probability','noise_std_range','blur_probability','blur_sigma_range'}
    if set(settings)!=expected: raise ValueError('Unknown/missing augmentation settings: '+str(set(settings)^expected))
    if type(settings['enabled']) is not bool: raise ValueError('augmentation.enabled must be boolean')
    def number(value): return type(value) in (int,float) and math.isfinite(value)
    for key in ('preserve_probability','noise_probability','blur_probability'):
        if not number(settings[key]) or not 0<=settings[key]<=1: raise ValueError('Invalid '+key)
    if not number(settings['brightness_delta']) or not 0<=settings['brightness_delta']<=.2:
        raise ValueError('brightness_delta must be in [0, 0.2] on normalized grayscale')
    for key,low,high in (('contrast_range',.5,1.5),('gamma_range',.5,1.5),('noise_std_range',0.,.05),('blur_sigma_range',.1,1.5)):
        values=settings[key]
        if not isinstance(values,(list,tuple)) or len(values)!=2 or not all(number(x) for x in values) or not low<=values[0]<=values[1]<=high:
            raise ValueError('Invalid light augmentation range: '+key)


class AppearanceAugmenter:
    def __init__(self,settings,seed):
        validate_augmentation_settings(settings)
        self.settings=copy.deepcopy(settings)
        self.rng=torch.Generator(device='cpu').manual_seed(seed)
        self.counts={domain:{key:0 for key in ('seen','kept','augmented','noise','blur','pixels_changed')}
            for domain in ('ir','satellite')}

    def uniform(self,low=0.,high=1.): return low+(high-low)*torch.rand((),generator=self.rng).item()

    def apply(self,u8,domain):
        if domain not in self.counts: raise ValueError('Explicit augmentation domain required')
        if u8.dtype!=np.uint8 or u8.ndim!=2: raise ValueError('Augmentation expects 2-D uint8 grayscale')
        c=self.settings; counts=self.counts[domain]; counts['seen']+=1
        metadata={'domain':domain,'applied':False}
        if not c['enabled'] or self.uniform()<c['preserve_probability']:
            counts['kept']+=1
            return u8.copy(),metadata
        brightness=self.uniform(-c['brightness_delta'],c['brightness_delta'])
        contrast=self.uniform(*c['contrast_range']); gamma=self.uniform(*c['gamma_range'])
        x=u8.astype(np.float32)/255.; center=float(x.mean())
        x=np.clip((x-center)*contrast+center+brightness,0.,1.)
        x=np.power(x,gamma)
        sigma=0.; noise=0.
        if self.uniform()<c['blur_probability']:
            sigma=self.uniform(*c['blur_sigma_range'])
            x=cv2.GaussianBlur(x,(5,5),sigmaX=sigma,borderType=cv2.BORDER_REFLECT_101); counts['blur']+=1
        if self.uniform()<c['noise_probability']:
            noise=self.uniform(*c['noise_std_range'])
            x=x+torch.randn(u8.shape,generator=self.rng).numpy()*noise; counts['noise']+=1
        result=np.rint(np.clip(x,0.,1.)*255.).astype(np.uint8)
        counts['augmented']+=1; counts['pixels_changed']+=int(np.count_nonzero(result!=u8))
        metadata.update(applied=True,brightness_delta=brightness,contrast=contrast,gamma=gamma,
            noise_std=noise,blur_sigma=sigma,mean_absolute_delta_u8=float(np.abs(result.astype(float)-u8).mean()))
        return result,metadata

    def state_dict(self):
        return {'settings_fingerprint':digest(self.settings),'rng_state':self.rng.get_state(),'counts':copy.deepcopy(self.counts)}

    def load_state_dict(self,state):
        if state['settings_fingerprint']!=digest(self.settings): raise ValueError('Augmentation resume settings mismatch')
        self.rng.set_state(state['rng_state']); self.counts=copy.deepcopy(state['counts'])


class AugmentedImages(OnlineImages):
    def __init__(self,manifest,settings,seed):
        super().__init__(manifest)
        self.augmentation=AppearanceAugmenter(settings,seed)

    def raw(self,group,index=None):
        c=self.config
        if index is None: u,_=query_image(group['image'],c['preprocess'])
        else:
            u=self.tiles.tile(group,index,c['preprocess']['size'])
            # image_bands [1,1,1] gives identical channels; true RGB maps fall back to grey for augmentation.
            if u.ndim==3: u=u[...,0].copy() if (u[...,0]==u[...,1]).all() and (u[...,0]==u[...,2]).all() else cv2.cvtColor(u,cv2.COLOR_BGR2GRAY)
        return u

    def read(self,group,index=None):
        # Evaluation and hard mining ALWAYS call this clean path, regardless of model.train().
        return tensor_image(self.raw(group,index),self.config['preprocess'])

    def read_train(self,group,index=None):
        if group['split']!='train': raise ValueError('Augmentation is restricted to training images')
        u,_=self.augmentation.apply(self.raw(group,index),'ir' if index is None else 'satellite')
        return tensor_image(u,self.config['preprocess'])


class TrainingViews:
    """Explicit view used ONLY by the loss forward, never mining or evaluation."""
    def __init__(self,images): self.images=images
    def read(self,group,index=None): return self.images.read_train(group,index)


def save_augmentation_preview(images,groups,output,seed):
    """Separate RNG: preview does not consume the training RNG or update counters."""
    out=Path(output)
    if (out/'augmentation_preview.json').exists(): return
    preview=AppearanceAugmenter(images.augmentation.settings,seed+7654321)
    chosen={}
    for group in groups:
        chosen.setdefault(group['group'],group)
    rows=[]
    for group in list(chosen.values())[:4]:
        positive=next((i for i,c in enumerate(group['candidates']) if c['label']==1),None)
        rows.append((group,None))
        if positive is not None: rows.append((group,positive))
    width=800; row_height=222
    canvas=Image.new('RGB',(width,60+row_height*len(rows)),(16,24,34)); draw=ImageDraw.Draw(canvas)
    label='SYNTHETIC | ' if images.manifest['synthetic'] else ''
    draw.text((12,8),label+'TRAIN-ONLY PREVIEW | independent preview RNG',fill='white')
    draw.text((12,28),'Original + 3 sampled views | coverage / orientation / labels unchanged',fill='white')
    entries=[]
    for row,(group,index) in enumerate(rows):
        domain='ir' if index is None else 'satellite'; original=images.raw(group,index)
        samples=[(original,{'original':True})]+[preview.apply(original,domain) for _ in range(3)]
        for col,(pixels,info) in enumerate(samples):
            x=col*200; y=60+row*row_height
            canvas.paste(Image.fromarray(pixels).resize((190,190)),(x+5,y))
            caption='original' if col==0 else ('augmented' if info['applied'] else 'kept original')
            draw.text((x+5,y+192),group['group']+' / '+domain+' / '+caption,fill='white')
            if col: entries.append({'id':group['id'],'candidate_index':index,'view':col,**info})
    temporary=out/'augmentation_preview.png.partial'; canvas.save(temporary,format='PNG')
    os.replace(temporary,out/'augmentation_preview.png')
    write_json(out/'augmentation_preview.json',{'training_rng_unchanged':True,'settings':preview.settings,
        'synthetic':images.manifest['synthetic'],'entries':entries})
