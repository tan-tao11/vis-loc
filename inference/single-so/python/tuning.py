"""Configuration resolution and per-worker tuning; upstream project files are untouched."""
import copy
import json
import math
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml

LEGACY = dict(initial_candidates=5, max_candidates=5, adaptive_candidates=False,
    coarse_resolution=560, refine_resolution=864, upsample=True, symmetric=True,
    sample_matches=3000, sample_threshold=.05, retrieval_overlap_percent=50,
    pre_homography_iterations=2000, post_homography_iterations=5000,
    homography_reprojection_px=3., geometry_confidence=.999,
    pnp_iterations=1000, pnp_reprojection_px=8., refine_max_nfev=None,
    early_min_inliers=40, early_max_reprojection_px=3., early_min_coverage=.12,
    early_agreement_m=30., early_require_agreement=True,
    min_result_inliers=5, reject_weak_results=False, resize_ratio=.3,
    scc_crop_gain=1.5, anyvis_patch_scale=1., search_window_m=600., profile_timing=True,
    track_max_candidates=0)   # >0: while tracking, match at most this many candidates (0 = max_candidates)


def merge(a,b):
    out=copy.deepcopy(a)
    for k,v in b.items():
        out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else copy.deepcopy(v)
    return out


def read_config(path):
    path=Path(path).resolve()
    raw=yaml.safe_load(path.read_text())
    config={k:copy.deepcopy(v) for k,v in raw.items() if k not in ('profiles','datasets')}
    profile=os.getenv('NAV_PROFILE',raw.get('active_profile',''))
    if profile:
        if profile not in raw.get('profiles',{}): raise ValueError(f'unknown profile: {profile}')
        config=merge(config,raw['profiles'][profile]);config['active_profile']=profile
    dataset=os.getenv('NAV_DATASET',raw.get('active_dataset',''))
    if dataset:
        if dataset not in raw.get('datasets',{}):raise ValueError(f'unknown dataset: {dataset}')
        config=merge(config,raw['datasets'][dataset]);config['active_dataset']=dataset
    config['tuning']=merge(LEGACY,config.get('tuning',{}))
    validate_tuning(config['tuning'])
    config['_config_dir']=str(path.parent)
    for key in ('rgb','infrared'):
        camera=config.get(key,{}).get('camera',{})
        if camera.get('frame_agl_file'):
            p=Path(camera['frame_agl_file'])
            if not p.is_absolute():p=path.parent/p
            camera['frame_agl_m']=json.loads(p.read_text())
    return config


def validate_tuning(p):
    unknown=set(p)-set(LEGACY)
    if unknown:raise ValueError(f'unknown tuning keys: {sorted(unknown)}')
    integer_bounds={'initial_candidates':(1,20),'max_candidates':(1,20),'coarse_resolution':(224,1120),
        'refine_resolution':(224,1792),'sample_matches':(128,10000),'pre_homography_iterations':(100,20000),
        'post_homography_iterations':(100,20000),'pnp_iterations':(100,20000),
        'early_min_inliers':(5,10000),'min_result_inliers':(5,10000),'track_max_candidates':(0,20)}
    for key,(lo,hi) in integer_bounds.items():
        v=p[key]
        if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi:raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
    if p['coarse_resolution']%14:raise ValueError('coarse_resolution must be a multiple of 14')
    if p['refine_resolution']%8:raise ValueError('refine_resolution must be a multiple of 8')
    if p['max_candidates']<p['initial_candidates']:raise ValueError('max_candidates < initial_candidates')
    if p['refine_resolution']<p['coarse_resolution']:raise ValueError('refine_resolution < coarse_resolution')
    if p['early_min_inliers']>p['sample_matches']:raise ValueError('early_min_inliers exceeds sample_matches')
    if p['refine_max_nfev'] is not None and (type(p['refine_max_nfev']) is not int or not 10<=p['refine_max_nfev']<=1000):
        raise ValueError('refine_max_nfev must be null or an integer in [10,1000]')
    for key,lo,hi in [('sample_threshold',.001,.99),('retrieval_overlap_percent',0,90),
        ('homography_reprojection_px',.1,20),('geometry_confidence',.9,.999999),
        ('pnp_reprojection_px',.1,30),('early_max_reprojection_px',.1,30),
        ('early_min_coverage',.001,1),('early_agreement_m',.1,1000),('resize_ratio',.05,1),
        ('scc_crop_gain',1,3),('anyvis_patch_scale',.5,3),('search_window_m',50,5000)]:
        v=p[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f'{key} must be in [{lo},{hi}]')
    for key in ('adaptive_candidates','upsample','symmetric','early_require_agreement','reject_weak_results','profile_timing'):
        if type(p[key]) is not bool:raise ValueError(f'{key} must be boolean')


def good_enough(records, p):
    good=[r for r in records if r.get('inliers',0)>=p['early_min_inliers'] and
          r.get('reprojection',float('inf'))<=p['early_max_reprojection_px'] and
          r.get('coverage',0)>=p['early_min_coverage'] and r.get('xy') is not None]
    if not good:return False
    if not p['early_require_agreement']:return True
    return any(np.linalg.norm(np.asarray(a['xy'])-b['xy'])<=p['early_agreement_m']
               for i,a in enumerate(good) for b in good[i+1:])


class CVProxy:
    """Change only this upstream utils module's calls, not the global OpenCV module."""
    def __init__(self, owner):self.owner=owner
    def __getattr__(self,key):return getattr(cv2,key)
    def findHomography(self,*args,**kwargs):
        p=self.owner.tuning
        args=list(args)
        if len(args)>=4:args[3]=p['homography_reprojection_px']
        else:kwargs['ransacReprojThreshold']=p['homography_reprojection_px']
        kwargs.update(maxIters=p['post_homography_iterations'],confidence=p['geometry_confidence'])
        return cv2.findHomography(*args,**kwargs)
    def solvePnPRansac(self,*args,**kwargs):
        p=self.owner.tuning
        kwargs.update(iterationsCount=p['pnp_iterations'],reprojectionError=p['pnp_reprojection_px'],confidence=p['geometry_confidence'])
        result=cv2.solvePnPRansac(*args,**kwargs)
        ok,rvec,tvec,inliers=result
        geometry=dict(inliers=0,reprojection=float('inf'),coverage=0.)
        if ok and inliers is not None and len(inliers):
            indices=np.asarray(inliers).reshape(-1)
            points=np.asarray(args[1]).reshape(-1,2)[indices]
            projected,_=cv2.projectPoints(np.asarray(args[0])[indices],rvec,tvec,args[2],args[3])
            h,w=self.owner.query_shape
            coverage=float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))/(w*h)
            geometry=dict(inliers=len(indices),reprojection=float(np.linalg.norm(points-projected.reshape(-1,2),axis=1).mean()),coverage=coverage)
        self.owner.last_geometry=geometry
        return result


def configure_matcher(owner, model):
    p=owner.tuning
    model.h_resized=model.w_resized=p['coarse_resolution']
    model.upsample_res=(p['refine_resolution'],p['refine_resolution'])
    model.upsample_preds=p['upsample'];model.symmetric=p['symmetric'];model.sample_thresh=p['sample_threshold']
    print('[NAV_CONFIG] '+json.dumps(dict(imgType=owner.mode,coarse_resolution=model.h_resized,
        refine_resolution=list(model.upsample_res),sample_matches=p['sample_matches'],
        initial_candidates=p['initial_candidates'],max_candidates=p['max_candidates'])),flush=True)


def match_pair(owner, a, b, model, need_ransac, convert_rgb):
    p=owner.tuning
    if convert_rgb:
        a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
    owner.torch.cuda.synchronize();start=time.perf_counter()
    with owner.torch.inference_mode():
        warp,certainty=model.match(Image.fromarray(a),Image.fromarray(b),device=owner.device)
        matches,confidence=model.sample(warp,certainty,num=p['sample_matches'])
        left,right=model.to_pixel_coordinates(matches,a.shape[0],a.shape[1],b.shape[0],b.shape[1])
        left,right,confidence=left.cpu().numpy(),right.cpu().numpy(),confidence.cpu().numpy()
    owner.torch.cuda.synchronize()
    owner.trace['dense_match_s']+=time.perf_counter()-start
    valid=np.isfinite(left).all(axis=1)&np.isfinite(right).all(axis=1)&np.isfinite(confidence)&(confidence>0)
    left,right,confidence=left[valid],right[valid],confidence[valid]
    order=np.argsort(confidence)[::-1];left,right,confidence=left[order],right[order],confidence[order]
    if owner.config.get("debug_matching", False):
        _debug_matching(owner, a, b, left, right, confidence, "before_ransac")
    if need_ransac and len(left)>=4:
        _,mask=cv2.findHomography(left,right,cv2.USAC_MAGSAC,p['homography_reprojection_px'],
                                  maxIters=p['pre_homography_iterations'],confidence=p['geometry_confidence'])
        if mask is not None:
            keep=mask.reshape(-1).astype(bool);left,right,confidence=left[keep],right[keep],confidence[keep]
    if owner.config.get("debug_matching", False):
        _debug_matching(owner, a, b, left, right, confidence, "after_ransac")
    return left.tolist(),right.tolist(),confidence.tolist()


def _debug_matching(owner, a, b, left, right, confidence, stage):
    try:
        frame = owner.trace.get('number', 0)
        candidate = owner.trace.get('candidates_attempted', 0)
        folder = Path(owner.config['_config_dir']) / 'debug_matching'
        folder = folder / f'mode{owner.mode}_frame{frame}_candidate{candidate}'
        folder.mkdir(parents=True, exist_ok=True)
        stem = f'{stage}_{time.time_ns()}'
        # a/b are the exact arrays passed to the matcher, before display resizing.
        Image.fromarray(a).save(folder / f'{stem}_a.png')
        Image.fromarray(b).save(folder / f'{stem}_b.png')
        np.savez_compressed(folder / f'{stem}_points.npz',
                            left=left, right=right, confidence=confidence)
        # Draw at most 150 points with the highest confidence.
        n = min(150, len(left))
        ka = [cv2.KeyPoint(float(x), float(y), 3) for x, y in left[:n]]
        kb = [cv2.KeyPoint(float(x), float(y), 3) for x, y in right[:n]]
        matches = [cv2.DMatch(i, i, 0.0) for i in range(n)]
        canvas = cv2.drawMatches(
            cv2.cvtColor(a, cv2.COLOR_RGB2BGR), ka,
            cv2.cvtColor(b, cv2.COLOR_RGB2BGR), kb, matches, None,
            matchColor=(0, 255, 0), flags=2)
        scale = min(1.0, 2000 / canvas.shape[1])
        if scale < 1:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
        if not cv2.imwrite(str(folder / f'{stem}_matches.jpg'), canvas):
            raise OSError('cannot write matching visualization')
        stats = dict(stage=stage, count=int(len(left)), drawn=n)
        (folder / f'{stem}_stats.json').write_text(json.dumps(stats))
        print(f'[MATCH_DEBUG] {stage} count={len(left)} folder={folder}', flush=True)
    except Exception as exc:
        print(f'[MATCH_DEBUG] save failed: {exc}', flush=True)
