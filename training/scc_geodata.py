"""Windowed TIFF + TFW + WKT reading. All transforms use GDAL pixel corners."""
import math
from pathlib import Path
import warnings
import cv2
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from affine import Affine
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform_bounds


def sidecar(image, suffixes):
    image = Path(image)
    names = {p.name.lower(): p for p in image.parent.iterdir() if p.is_file()}
    for suffix in suffixes:
        p = names.get(image.with_suffix(suffix).name.lower())
        if p: return p
    return None


def read_text(path):
    raw = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
        try: return raw.decode(encoding).strip()
        except UnicodeError: pass
    raise ValueError(f'cannot decode projection/world file: {path}')


def world_transform(path):
    values = [float(v) for v in read_text(path).split()]
    if len(values) != 6 or not np.isfinite(values).all():
        raise ValueError('TFW must contain six finite numbers: ' + str(path))
    a, d, b, e, c, f = values
    t = Affine(a, b, c-(a+b)*.5, d, e, f-(d+e)*.5)
    if abs(t.determinant) < 1e-15: raise ValueError('singular TFW transform')
    return t


def metadata(src, ref, role):
    tfw = ref.get(role+'_tfw') or sidecar(src.name, ('.tfw', '.tifw', '.wld'))
    prj = ref.get(role+'_prj') or ref.get('crs_file') or sidecar(src.name, ('.prj', '.wkt'))
    embedded = src.crs
    crs = CRS.from_user_input(read_text(prj) if prj else (embedded or ref.get(role+'_crs') or ref.get('crs')))
    if embedded and not crs.equals(CRS.from_user_input(embedded), ignore_axis_order=True):
        raise ValueError(f'{role}: embedded CRS and sidecar/config CRS disagree')
    explicit = ref.get(role+'_crs') or ref.get('crs')
    if explicit and not CRS.from_user_input(explicit).equals(crs, ignore_axis_order=True):
        raise ValueError(f'{role}: configured CRS disagrees with file CRS')
    transform = world_transform(tfw) if tfw else src.transform
    if tfw and src.transform != Affine.identity() and not np.allclose(tuple(transform), tuple(src.transform), atol=1e-6, rtol=0):
        raise ValueError(f'{role}: embedded transform and TFW disagree')
    if not tfw and src.transform == Affine.identity():
        raise ValueError(f'{role}: no geotransform; supply GeoTIFF or a matching TFW')
    if not np.isfinite(tuple(transform)).all() or abs(transform.determinant) < 1e-15:
        raise ValueError(f'{role}: invalid affine transform')
    return crs, transform


class GeoReference:
    def __init__(self, ref):
        self.ref = dict(ref)
        self.sources = []
        self.vrts = []
        self.env = rasterio.Env(GDAL_CACHEMAX=64*1024*1024, PROJ_NETWORK='OFF')
        self.env.__enter__()
        try:
            for role in ('image', 'dsm'):
                src = rasterio.open(ref[role])
                self.sources.append(src)
            image, dsm = self.sources
            image_crs, image_t = metadata(image, ref, 'image')
            dsm_crs, dsm_t = metadata(dsm, ref, 'dsm')
            working = CRS.from_user_input(ref.get('working_crs') or dsm_crs)
            if not working.is_projected or any(abs(a.unit_conversion_factor-1)>1e-10 for a in working.axis_info[:2]):
                raise ValueError('DSM/working_crs must be a projected CRS in metres')
            self.bands = list(ref.get('image_bands', [1, 2, 3]))
            if len(self.bands)!=3 or min(self.bands)<1 or max(self.bands)>image.count:
                raise ValueError('image_bands must select three existing RGB bands')
            self.dsm_band = int(ref.get('dsm_band', 1))
            if not 1 <= self.dsm_band <= dsm.count: raise ValueError('invalid dsm_band')
            self.value_range = ref.get('image_value_range')
            if any(image.dtypes[i-1] != 'uint8' for i in self.bands) and self.value_range is None:
                raise ValueError('non-uint8 reference requires explicit image_value_range: [min,max]')
            if self.value_range is not None:
                lo, hi = map(float, self.value_range)
                if not (math.isfinite(lo) and math.isfinite(hi) and hi>lo): raise ValueError('invalid image_value_range')
            corners = [dsm_t*(c,r) for c,r in ((0,0),(dsm.width,0),(0,dsm.height),(dsm.width,dsm.height))]
            xs,ys = zip(*corners)
            bounds = (min(xs),min(ys),max(xs),max(ys))
            same_crs = working.equals(dsm_crs, ignore_axis_order=True)
            if not same_crs: bounds=transform_bounds(dsm_crs.to_wkt(),working.to_wkt(),*bounds,densify_pts=21)
            if ref.get('resolution_m') is not None: self.res=float(ref['resolution_m'])
            elif same_crs: self.res=min(math.hypot(dsm_t.a,dsm_t.d),math.hypot(dsm_t.b,dsm_t.e))
            else: raise ValueError('resolution_m is required when changing the DSM CRS')
            if not math.isfinite(self.res) or self.res<=0: raise ValueError('invalid output resolution_m')
            left,bottom,right,top=bounds
            self.x0,self.y0=left+self.res*.5,top-self.res*.5
            self.shape=(max(1,math.ceil((top-bottom)/self.res)),max(1,math.ceil((right-left)/self.res)))
            self.grid=Affine(self.res,0,left,0,-self.res,top)
            for role,src,crs,t in [('image',image,image_crs,image_t),('dsm',dsm,dsm_crs,dsm_t)]:
                options=dict(src_crs=crs.to_wkt(),src_transform=t,crs=working.to_wkt(),transform=self.grid,
                             width=self.shape[1],height=self.shape[0],resampling=Resampling.bilinear,
                             warp_mem_limit=64,dtype='float32',nodata=float('nan'))
                if role=='dsm' and ref.get('nodata') is not None:options['src_nodata']=float(ref['nodata'])
                self.vrts.append(WarpedVRT(src,**options))
            self.dsm_scale=float(dsm.scales[self.dsm_band-1]);self.dsm_offset=float(dsm.offsets[self.dsm_band-1])
            self.forward=Transformer.from_crs('EPSG:4326',working,always_xy=True)
            self.inverse=Transformer.from_crs(working,'EPSG:4326',always_xy=True)
            self.window_m=float(ref.get('window_m',600))
            self.image=self.dsm=None;self.z0=0.
            if self.window_m<=0:
                if self.shape[0]*self.shape[1]>25000000:raise ValueError('large raster requires window_m')
                view=self.read_window(0,self.shape[0],0,self.shape[1])
                self.image,self.dsm,self.z0=view.image,view.dsm,view.z0
        except Exception:
            self.close()
            raise

    def close(self):
        for obj in getattr(self,'vrts',[]):obj.close()
        for obj in getattr(self,'sources',[]):obj.close()
        if getattr(self,'env',None) is not None:
            self.env.__exit__(None,None,None);self.env=None

    def __del__(self):
        try:self.close()
        except Exception:pass

    def read_window(self,r0,r1,c0,c1):
        window=Window(c0,r0,c1-c0,r1-r0)
        rgb=self.vrts[0].read(self.bands,window=window,masked=True).filled(np.nan).transpose(1,2,0)
        dsm=self.vrts[1].read(self.dsm_band,window=window,masked=True).filled(np.nan)
        dsm=dsm*self.dsm_scale+self.dsm_offset
        valid=np.isfinite(dsm)&np.isfinite(rgb).all(axis=2)
        if not valid.any():raise ValueError('no overlapping valid map/DSM pixels')
        if self.value_range is not None:
            lo,hi=map(float,self.value_range);rgb=(rgb-lo)*(255/(hi-lo))
        rgb=np.clip(np.nan_to_num(rgb),0,255).astype(np.uint8)
        image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);image[~valid]=0;dsm[~valid]=np.nan
        view=GeoReference.__new__(GeoReference)
        view.image,view.dsm=image,dsm
        view.x0,view.y0=self.x0+c0*self.res,self.y0-r0*self.res
        view.res=self.res;view.forward=self.forward;view.inverse=self.inverse
        view.z0=float(np.nanmedian(dsm));view.shape=dsm.shape;view.window_m=0
        return view

    def for_frame(self,t):
        if self.window_m<=0:return self
        x,y=self.local_prior(t)
        if not (0<=x/self.res<self.shape[1] and 0<=y/self.res<self.shape[0]):
            raise ValueError('position prior outside reference coverage')
        side=max(32,int(round(self.window_m/self.res)))
        c0=max(0,min(int(np.floor(x/self.res-side/2)),max(0,self.shape[1]-side)))
        r0=max(0,min(int(np.floor(y/self.res-side/2)),max(0,self.shape[0]-side)))
        return self.read_window(r0,min(r0+side,self.shape[0]),c0,min(c0+side,self.shape[1]))

    def local_prior(self,t):
        x,y=self.forward.transform(t['lon'],t['lat'],errcheck=True)
        return x-self.x0,self.y0-y

    def agl(self,t,camera):
        if 'agl_m' in camera:h=float(camera['agl_m'])
        else:
            x,y=self.local_prior(t);col,row=int(round(x/self.res)),int(round(y/self.res))
            if not(0<=row<self.dsm.shape[0] and 0<=col<self.dsm.shape[1]):raise ValueError('position prior outside DSM')
            h=t['altitude']-float(self.dsm[row,col])
        if not math.isfinite(h) or h<=0:raise ValueError('camera height above terrain must be positive')
        return h

    def local_to_geographic(self,xyz):
        lon,lat=self.inverse.transform(self.x0+xyz['X'],self.y0-xyz['Y'],errcheck=True)
        return lon,lat,xyz['Z']+self.z0
