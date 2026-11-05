"""Start packaged CPython with its own site-packages; never use host site-packages."""
import os
from pathlib import Path
import runpy
import site
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTHONNOUSERSITE']='1'
os.environ['MPLBACKEND']='Agg'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PROJ_NETWORK']='OFF'
os.environ['HF_HOME']=str(ROOT/'cache/huggingface')
os.environ['TORCH_HOME']=str(ROOT/'cache/torch')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
site.addsitedir(str(ROOT/'env/lib/python3.10/site-packages'))
site.addsitedir(str(ROOT/'extra'))
sys.path.insert(0,str(ROOT/'adapter'))
os.environ['NAV_EMBEDDED_ROOT']=str(ROOT)
if '--self-test' in sys.argv:
    import torch, torchvision, rasterio, pyproj, scipy, cv2
    assert torch.cuda.is_available()
    with torch.inference_mode():
        a=torch.eye(8,device='cuda')
        assert torch.allclose(torch.linalg.inv(a),a)
        assert torch.nn.Conv2d(3,4,3).cuda()(torch.zeros(1,3,8,8,device='cuda')).isfinite().all()
        assert torchvision.ops.nms(torch.tensor([[0,0,2,2]],device='cuda',dtype=torch.float32),torch.ones(1,device='cuda'),.5).numel()==1
    print('BUNDLED_RUNTIME_OK',sys.version,torch.__version__,rasterio.__version__,sys.executable,flush=True)
elif '--script' in sys.argv:
    i=sys.argv.index('--script');script=sys.argv[i+1];sys.argv=[script]+sys.argv[i+2:]
    runpy.run_path(script,run_name='__main__')
else:
    runpy.run_path(str(ROOT/'adapter/worker.py'),run_name='__main__')
