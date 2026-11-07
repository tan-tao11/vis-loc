"""Resolve bundled code and externally configured weights without changing model math."""
import os
from pathlib import Path
from urllib.parse import urlparse

NAMES={
    'dinov2':'dinov2_vitl14_pretrain.pth',
    'roma':'roma_outdoor.pth',
    'minima_roma':'minima_roma.pth',
}

def configure(config,modality):
    root=Path(os.environ['NAV_EMBEDDED_ROOT'])
    config.setdefault('infrared',{})['project_root']=str(root/'projects'/'SCC-Loc')
    if modality!=0:return
    base=Path(config['_config_dir'])
    w=config.get('weights',{})
    weight_root=Path(w.get('root',str(base/'weights')))
    if not weight_root.is_absolute():weight_root=base/weight_root
    needed=['dinov2','roma','minima_roma']
    paths={}
    for key in needed:
        path=Path(w.get(key,str(weight_root/NAMES[key])))
        if not path.is_absolute():path=base/path
        if not path.is_file():raise ValueError(f'missing external weight {key}: {path}')
        paths[NAMES[key]]=str(path.resolve())
    import torch
    original=torch.load
    def load(f,*args,**kwargs):
        if isinstance(f,(str,os.PathLike)):
            name=Path(f).name
            if name in paths:f=paths[name]
            elif name in NAMES.values():raise ValueError(f'unconfigured external weight: {name}')
        return original(f,*args,**kwargs)
    torch.load=load
    def load_url(url,model_dir=None,map_location=None,progress=True,check_hash=False,file_name=None,weights_only=False):
        name=file_name or Path(urlparse(url).path).name
        if name not in paths:raise ValueError(f'offline inference: weight URL not configured: {name}')
        return original(paths[name],map_location=map_location,weights_only=weights_only)
    torch.hub.load_state_dict_from_url=load_url
