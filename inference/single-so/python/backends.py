"""Adapters reuse the deployed retrieval, matching and pose-estimation implementations.

All maps use a shared explicit north-up, projected, metre-based raster contract.
Map origin means the centre of pixel (0, 0), not its upper-left corner.
DSM must already be aligned with the reference image and use normal heights (m).
"""
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from pyproj import CRS, Transformer
from tuning import read_config, merge, validate_tuning, CVProxy, configure_matcher, match_pair, good_enough
from prior_tracker import PriorTracker
from oblique import view_centre, rectify_query
import consistency
import self_eval

# Output sources (see localize_tracked) reported as a successful localization; others make stepVNavigator return 0.
REPORT_ALL = ('fix', 'unconfirmed_fix', 'predicted', 'retrieval', 'search_centre')


class NotReported(RuntimeError):
    """The frame has a position but is configured to be reported as not localized."""


TELEMETRY_KEYS = ("lon", "lat", "altitude", "drone_yaw", "drone_pitch", "drone_roll",
                  "sensor_yaw", "sensor_pitch", "sensor_roll", "inns_yaw", "inns_pitch",
                  "inns_roll", "hfov", "vfov", "ccd_size")


def absolute_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference image/DSM paths must be filled in naviParams.txt")
    p = Path(value).expanduser()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def load_config(path):
    path = Path(path).resolve()
    config = read_config(path)
    if os.environ.get('NAV_EMBEDDED_ROOT'):
        embedded = Path(os.environ['NAV_EMBEDDED_ROOT'])/'projects'
        for key, project in [('rgb','AnyVisLoc-Benchmark'),('infrared','SCC-Loc')]:
            config.setdefault(key,{})['project_root']=str(embedded/project)
    if config.get("version") != 1:
        raise ValueError("unsupported config version")
    enabled = config.get("enabled_modalities", [0, 1])
    if not enabled or len(set(enabled)) != len(enabled) or any(m not in (0, 1) for m in enabled):
        raise ValueError("enabled_modalities must contain 0 and/or 1")
    for mode in enabled:
        key = "rgb" if mode == 1 else "infrared"
        section = config[key]
        section["project_root"] = str(absolute_path(section["project_root"], path.parent))
        if not Path(section["project_root"]).is_dir():
            raise ValueError(f"{key}: missing project_root")
        ref = section["reference"]
        if config.get('active_profile'):
            ref['window_m']=merge(config['tuning'],section.get('tuning',{}))['search_window_m']
        for field in ("image", "dsm"):
            ref[field] = str(absolute_path(ref[field], path.parent))
            if not Path(ref[field]).is_file():
                raise ValueError(f"{key}: missing {field}: {ref[field]}")
        if section.get('retrieval_head'):
            section['retrieval_head'] = str(absolute_path(section['retrieval_head'], path.parent))
            if not Path(section['retrieval_head']).is_file():
                raise ValueError(f"{key}: missing retrieval_head: {section['retrieval_head']}")
        for field in ('image_tfw','dsm_tfw','image_prj','dsm_prj','crs_file'):
            if ref.get(field):ref[field]=str(absolute_path(ref[field],path.parent))
        if ref.get("height_datum") != "normal_m" and not (
            config.get("allow_unverified_height_datum", False) and ref.get("height_datum") == "unverified_m"):
            raise ValueError(f"{key}: DSM must use normal heights in metres, height_datum=normal_m")
        if ref.get('georeference') == 'auto':
            continue  # TIFF/TFW/WKT validation is performed by GeoReference.
        if any(ref.get(k) is None for k in ("crs", "origin_xy", "resolution_m")):
            raise ValueError(f"{key}: fill actual crs, origin_xy and resolution_m; no default georeferencing is inferred")
        crs = CRS.from_user_input(ref["crs"])
        if not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-10 for a in crs.axis_info[:2]):
            raise ValueError(f"{key}: reference CRS must be projected with metre units")
        origin = np.asarray(ref["origin_xy"], dtype=float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError(f"{key}: invalid origin_xy")
        resolution = float(ref["resolution_m"])
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"{key}: resolution_m must be positive")
        if section.get("pose_priori", "unknown") not in ("unknown", "yp"):
            raise ValueError("pose_priori must be unknown or yp")
        if not 0.05 <= float(section.get("resize_ratio", .4)) <= 1:
            raise ValueError("resize_ratio must be in [0.05, 1]")
        if not 1 <= int(section.get("top_n", 5)) <= 100:
            raise ValueError("top_n must be in [1, 100]")
    return config


class WindowRaster:
    """Read only selected rows/columns; supports this dataset's unordered TIFF strips."""
    def __init__(self, filename, image=False):
        self.fd = None
        self.array = None
        self.rgb = False
        p = Path(filename)
        if p.suffix.lower() in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(p) as tif:
                page = tif.pages[0]
                self.shape = page.shape
                self.dtype = np.dtype(tif.byteorder + page.dtype.str[1:])
                self.rgb = image and len(self.shape) == 3
                if page.is_memmappable:
                    self.array = tifffile.memmap(p)
                elif (page.compression == 1 and page.rowsperstrip == 1 and
                      len(page.dataoffsets) == self.shape[0] and page.predictor == 1 and
                      page.planarconfig == 1):
                    self.offsets = page.dataoffsets
                    self.fd = os.open(p, os.O_RDONLY)
                else:
                    raise ValueError("unsupported TIFF layout for window reads; convert to uncompressed strips")
        elif p.suffix.lower() == ".npy":
            self.array = np.load(p, mmap_mode="r", allow_pickle=False)
        else:
            self.array = cv2.imread(str(p), cv2.IMREAD_COLOR if image else cv2.IMREAD_UNCHANGED)
            if self.array is None:
                raise ValueError(f"cannot read raster: {p}")
        if self.array is not None:
            self.shape = self.array.shape
            self.dtype = self.array.dtype

    def __del__(self):
        if self.fd is not None:
            os.close(self.fd)

    def read(self, r0, r1, c0, c1):
        if self.array is not None:
            return np.array(self.array[r0:r1, c0:c1], copy=True)
        channels = self.shape[2] if len(self.shape) == 3 else 1
        out_shape = (r1-r0, c1-c0, channels) if len(self.shape) == 3 else (r1-r0, c1-c0)
        out = np.empty(out_shape, self.dtype)
        count = (c1-c0)*channels
        for j, row in enumerate(range(r0, r1)):
            raw = os.pread(self.fd, count*self.dtype.itemsize,
                           self.offsets[row]+c0*channels*self.dtype.itemsize)
            if len(raw) != count*self.dtype.itemsize:
                raise IOError("truncated TIFF strip")
            out[j] = np.frombuffer(raw, self.dtype).reshape(out[j].shape)
        return out


class Reference:
    def __new__(cls, ref=None):
        if ref is not None and ref.get('georeference') == 'auto':
            from geodata import GeoReference
            return GeoReference(ref)
        return super().__new__(cls)

    def __init__(self, ref):
        self.source_image = WindowRaster(ref["image"], image=True)
        self.source_dsm = WindowRaster(ref["dsm"])
        self.shape = self.source_dsm.shape
        if len(self.shape) != 2 or min(self.shape) < 32 or len(self.source_image.shape) != 3:
            raise ValueError("expected 2D DSM and three-channel reference")
        if self.source_image.shape[2] != 3:
            raise ValueError("reference image must have three channels")
        self.x0, self.y0 = map(float, ref["origin_xy"])
        self.res = float(ref["resolution_m"])
        self.image_res = np.asarray(ref.get("image_resolution_xy", [self.res, self.res]), dtype=float)
        self.image_origin = np.asarray(ref.get("image_origin_xy", [self.x0, self.y0]), dtype=float)
        if self.image_res.shape != (2,) or not np.isfinite(self.image_res).all() or np.any(self.image_res <= 0):
            raise ValueError("invalid image_resolution_xy")
        self.nodata = ref.get("nodata")
        self.window_m = float(ref.get("window_m", 0))
        self.forward = Transformer.from_crs("EPSG:4326", ref["crs"], always_xy=True)
        self.inverse = Transformer.from_crs(ref["crs"], "EPSG:4326", always_xy=True)
        if self.window_m > 0:
            self.image = self.dsm = None
            self.z0 = 0.
        else:
            if self.shape[0]*self.shape[1] > 25000000:
                raise ValueError("large reference requires reference.window_m for local search")
            view = self.read_window(0, self.shape[0], 0, self.shape[1])
            self.image, self.dsm, self.z0 = view.image, view.dsm, view.z0

    def read_window(self, r0, r1, c0, c1):
        dsm = self.source_dsm.read(r0,r1,c0,c1).astype(np.float32,copy=False)
        if self.nodata is not None:
            dsm[dsm == float(self.nodata)] = np.nan
        if not np.isfinite(dsm).any():
            raise ValueError("selected window has no valid DSM heights")
        # Resample the RGB raster onto the DSM grid using actual pixel-centre georeferencing.
        cols=(self.x0+np.arange(c0,c1)*self.res-self.image_origin[0])/self.image_res[0]
        rows=(self.image_origin[1]-(self.y0-np.arange(r0,r1)*self.res))/self.image_res[1]
        h,w=self.source_image.shape[:2]
        ir0=max(0,int(np.floor(rows.min()))); ir1=min(h,int(np.ceil(rows.max()))+2)
        ic0=max(0,int(np.floor(cols.min()))); ic1=min(w,int(np.ceil(cols.max()))+2)
        if ir0>=ir1 or ic0>=ic1:
            raise ValueError("no reference image coverage for selected DSM window")
        image=self.source_image.read(ir0,ir1,ic0,ic1)
        if self.source_image.rgb:
            image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        xx,yy=np.meshgrid(cols-ic0,rows-ir0)
        image=cv2.remap(image,xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        valid=(xx>=0)&(xx<=ic1-ic0-1)&(yy>=0)&(yy<=ir1-ir0-1)&np.isfinite(dsm)
        image[~valid]=0
        dsm[~valid]=np.nan
        if not np.isfinite(dsm).any():
            raise ValueError("no overlapping valid map/DSM pixels")
        view=Reference.__new__(Reference)
        view.image,view.dsm=image,dsm
        view.x0,view.y0=self.x0+c0*self.res,self.y0-r0*self.res
        view.res=self.res;view.forward=self.forward;view.inverse=self.inverse
        view.z0=float(np.nanmedian(dsm));view.shape=dsm.shape
        return view

    def for_frame(self, telemetry):
        if self.window_m <= 0:
            return self
        x,y=self.local_prior(telemetry)
        side=max(32,int(round(self.window_m/self.res)))
        c0=int(np.floor(x/self.res-side/2)); r0=int(np.floor(y/self.res-side/2))
        r0=max(0,min(r0,max(0,self.shape[0]-side)))
        c0=max(0,min(c0,max(0,self.shape[1]-side)))
        return self.read_window(r0,min(r0+side,self.shape[0]),c0,min(c0+side,self.shape[1]))

    def local_prior(self, telemetry):
        x, y = self.forward.transform(telemetry["lon"], telemetry["lat"], errcheck=True)
        return x - self.x0, self.y0 - y

    def agl(self, telemetry, camera):
        if "agl_m" in camera:
            h = float(camera["agl_m"])
        else:
            x, y = self.local_prior(telemetry)
            col, row = int(round(x / self.res)), int(round(y / self.res))
            if not (0 <= row < self.dsm.shape[0] and 0 <= col < self.dsm.shape[1]):
                raise ValueError("position prior outside DSM; supply a valid prior or explicit camera.agl_m")
            h = telemetry["altitude"] - float(self.dsm[row, col])
        if not math.isfinite(h) or h <= 0:
            raise ValueError("camera height above terrain must be positive")
        return h

    def local_to_geographic(self, xyz):
        lon, lat = self.inverse.transform(self.x0 + xyz["X"], self.y0 - xyz["Y"], errcheck=True)
        return lon, lat, xyz["Z"] + self.z0


def camera_matrix(camera, t, width, height):
    if "K" in camera:
        K = np.asarray(camera["K"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("invalid camera K")
        if not np.allclose(K[2], [0, 0, 1]):
            raise ValueError("invalid camera K last row")
        calibrated_w, calibrated_h = map(int, camera["image_size"])
        if min(calibrated_w, calibrated_h) <= 0:
            raise ValueError("invalid calibration image_size [width,height]")
        K = np.diag([width/calibrated_w, height/calibrated_h, 1]) @ K
    else:
        hfov, vfov = t["hfov"], t["vfov"]
        if not (0 < hfov < 179 and 0 < vfov < 179):
            raise ValueError("valid hfov/vfov or calibrated camera.K is required")
        K = np.array([[width/(2*math.tan(math.radians(hfov)/2)), 0, (width-1)/2],
                      [0, height/(2*math.tan(math.radians(vfov)/2)), (height-1)/2], [0, 0, 1]], dtype=float)
    return K


def optical_pose(t, is_eo, camera):
    """Body-to-NED Rz(yaw)Ry(pitch)Rx(roll), with an explicit optical axis in INS axes."""
    # Conversation-reconstructed near-nadir reference alignment, not full extrinsics.
    # The 1-degree gates are approximation checks, not measured calibration accuracy.
    if camera.get("nadir_reference_alignment", False):
        if is_eo:
            if abs(t["inns_pitch"] + 90.0) > 1.0:
                raise ValueError("EO nadir alignment requires innsPitch near -90; disable it for slant")
            return (t["inns_yaw"] + t["inns_roll"]) % 360.0, -90.0
        if any(abs(t[k]) > 1.0 for k in (
                "sensor_yaw", "sensor_pitch", "sensor_roll", "inns_pitch", "inns_roll")):
            raise ValueError("non-EO nadir alignment requires near-zero sensor angles and INS pitch/roll")
        return t["inns_yaw"] % 360.0, -90.0
    if "optical_axis_in_inns" in camera:
        axis = np.asarray(camera["optical_axis_in_inns"], dtype=float)
    elif is_eo:
        axis = np.array([1., 0., 0.])  # Interface explicitly states EO optical axis aligns with INS X.
    else:
        raise ValueError("yp prior for non-EO needs calibrated optical_axis_in_inns")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("invalid optical_axis_in_inns")
    yaw, pitch, roll = np.deg2rad([t["inns_yaw"], t["inns_pitch"], t["inns_roll"]])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                  [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
    n, e, d = R @ (axis / np.linalg.norm(axis))
    elevation = math.degrees(math.atan2(-d, math.hypot(n, e)))
    if elevation >= -1:
        raise ValueError("optical axis must point below the horizon for map localization")
    return math.degrees(math.atan2(e, n)) % 360, elevation


class BaseBackend:
    def __init__(self, config, modality):
        self.mode = modality
        self.config = config
        self.section = config["rgb" if modality == 1 else "infrared"]
        self.tuning = merge(config.get('tuning',{}),self.section.get('tuning',{}))
        if not config.get('active_profile'):
            # Existing saved benchmark configs retain their requested candidate count.
            if 'top_n' in self.section:
                self.tuning['initial_candidates']=self.tuning['max_candidates']=int(self.section['top_n'])
            self.tuning['resize_ratio']=float(self.section.get('resize_ratio',self.tuning['resize_ratio']))
        validate_tuning(self.tuning)
        reference=dict(self.section['reference'])
        if config.get('active_profile'):reference['window_m']=self.tuning['search_window_m']
        self.ref = Reference(reference)
        self.full_ref = self.ref
        # Large independent per-frame prior errors: smooth over the sequence, then track accepted fixes.
        tracking = self.section.get('prior_tracking', {})
        self.tracker = PriorTracker(tracking) if tracking.get('enabled', False) and self.full_ref.window_m > 0 else None
        self.check = consistency.settings(self.section.get('consistency'))
        self.last_fix = None
        self.camera = self.section.get("camera", {})
        self.device = config.get("device", "cuda")
        if self.device not in ("cuda", "cuda:0"):
            raise ValueError("these deployed model wrappers require cuda:0")
        root = self.section["project_root"]
        os.chdir(root)
        sys.path.insert(0, root)
        sys.argv = [sys.argv[0]]  # Legacy model loaders parse argv themselves.
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in isolated environment")
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
        torch.manual_seed(42)
        self.pose = self.section.get("pose_priori", "unknown")
        self.settings = dict(DEVICE=self.device, BATCH_SIZE=int(config.get("batch_size", 8)),
                             RETRIEVAL_COVER=self.tuning['retrieval_overlap_percent'], RETRIEVAL_TOPN=1,
                             RETRIEVAL_IMG="/retrieval.png", RETRIEVAL_FEATURE_NORM=True,
                             SHOW_RETRIEVAL_RESULT=False, SHOW_RETRIEVAL_IMG_NUM=5, USE_DEM=True)

    def begin_frame(self, request):
        self.frame_start=time.perf_counter()
        self.trace=dict(number=request['number'],imgType=self.mode,profile=self.config.get('active_profile','legacy'),
                        dense_match_s=0.,pose_s=0.,retrieval_s=0.,prepare_s=0.,candidate_pipeline_s=0.,
                        candidates_attempted=0,candidates=[],fallback_used=False)
        self.last_fix=None; self.last_source=None

    def end_frame(self, success, error=None):
        self.trace.update(success=success,error=error,total_s=time.perf_counter()-self.frame_start)
        self.trace['filter_and_glue_s']=max(0.,self.trace['candidate_pipeline_s']-self.trace['dense_match_s']-self.trace['pose_s'])
        if not self.tuning['profile_timing']:return
        def clean(v):
            if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [clean(x) for x in v]
            if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
            if isinstance(v,np.generic):return v.item()
            return v
        text=json.dumps(clean(self.trace),allow_nan=False)
        print('[NAV_TIMING] '+text,flush=True)
        logfile=self.config.get('timing_file')
        if logfile:
            p=Path(logfile)
            if not p.is_absolute():p=Path(self.config['_config_dir'])/p
            try:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('a') as f:f.write(text+'\n')
            except OSError as exc:print(f'[NAV_TIMING] cannot write log: {exc}',file=sys.stderr)

    def self_eval_cfg(self):
        return self_eval.settings(getattr(self, 'section', {}).get('self_eval'))

    def report_success(self, source):
        """Raise NotReported if this output source is configured as a failed frame (stepVNavigator returns 0)."""
        allowed = getattr(self, 'section', {}).get('report_success', REPORT_ALL)
        self.trace.update(output_source=source, reported=source in allowed)
        if source not in allowed:
            raise NotReported('frame not reported as localized: source=%s' % source)

    def localize_tracked(self, request, image):
        """localize() plus prior tracking: accept/reject the fix and choose what to report.

        Without prior_tracking this is exactly localize(). With it, a failed or untrusted frame while
        tracking reports the predicted position (confidence 0) instead of failing or a wrong match.
        """
        tracker=getattr(self,'tracker',None)
        if tracker is None:
            result=self.localize(request, image)
            source='retrieval' if getattr(self,'last_source',None)=='retrieval' else 'fix'
            if self.self_eval_cfg()['enabled']:
                value,failed=self_eval.decide(self.trace.get('self_eval_features'),source,None,self.self_eval_cfg())
                result=tuple(result[:3])+(value,); self.trace.update(self_eval=value,self_eval_failed=failed)
            self.trace['output']=list(result)
            self.report_success(source)
            return result
        try:
            result=self.localize(request, image); error=None
        except Exception as exc:
            result=None; error=exc; import traceback; traceback.print_exc()
        number=int(request["number"])
        if tracker.last is None or tracker.last['number']!=number:
            if error is not None: raise error           # failed before this frame's prediction
            return result
        x=y=None; accepted=False
        if result is not None:
            x,y=self.full_ref.forward.transform(result[0],result[1],errcheck=True)
            if getattr(self,'last_fix',None) is not None:   # only checked PnP fixes update the tracker
                accepted=tracker.update(number,x,y,self.last_fix[2])
        state=dict(tracker.last)
        residual=math.hypot(x-state['x'],y-state['y']) if (x is not None and state['mode']=='track') else None
        ox,oy,source=tracker.output(x,y,accepted)
        if source=='unconfirmed_fix' and getattr(self,'last_source',None)=='retrieval': source='retrieval'
        if accepted: self.last_output_altitude=float(result[2])
        if source in ('fix','unconfirmed_fix','retrieval'):
            out=result
        else:
            lon,lat=self.full_ref.inverse.transform(ox,oy,errcheck=True)
            alt=getattr(self,'last_output_altitude',None)
            out=(lon,lat,float(tracker.last['altitude'] if alt is None else alt),0)
        self.trace.update(prior_fix_accepted=accepted,output_source=source,
                          localize_error=None if error is None else str(error))
        cfg=self.self_eval_cfg()
        if cfg['enabled']:   # disabled: keep the backend's own value (0 for SCC, AnyVis confidence)
            value,failed=self_eval.decide(self.trace.get('self_eval_features'),source,residual,cfg)
            out=tuple(out[:3])+(value,)
        else:
            value,failed=int(out[3]),['disabled']
        self.trace.update(self_eval=value,self_eval_failed=failed,track_residual_m=residual,output=list(out))
        allowed=getattr(self,'section',{}).get('report_success',REPORT_ALL)
        print('[SELF_EVAL] frame=%s value=%d failed=%s'%(number,value,','.join(failed) or '-'),flush=True)
        print('[PRIOR_OUT] frame=%s source=%s accepted=%s mode=%s reported=%d'%(number,source,accepted,tracker.last['mode'],source in allowed),flush=True)
        self.report_success(source)
        return out

    def candidate_finished(self, xy, count, reprojection=None):
        record=dict(self.last_geometry)
        record.update(xy=xy,inliers=int(count))
        if reprojection is not None:record['reprojection']=float(reprojection)
        self.trace['candidates'].append(record)
        n=self.trace['candidates_attempted']
        if n<self.tuning['initial_candidates']:return False
        stop=self.tuning['adaptive_candidates'] and good_enough(self.trace['candidates'],self.tuning)
        return stop

    def prepare(self, request, image):
        if request["imgType"] != self.mode:
            raise ValueError("worker modality mismatch")
        vals = request["telemetry"]
        if len(vals) != len(TELEMETRY_KEYS) or not np.isfinite(vals).all():
            raise ValueError("invalid telemetry")
        t = dict(zip(TELEMETRY_KEYS, map(float, vals)))
        if abs(t["lon"]) > 180 or abs(t["lat"]) > 90:
            raise ValueError("invalid position prior")
        h, w = image.shape[:2]
        self.query_shape=(h,w)
        K = camera_matrix(self.camera, t, w, h)
        camera = dict(self.camera)
        frame_agl = camera.get("frame_agl_m", {})
        if str(request["number"]) in frame_agl:
            camera["agl_m"] = frame_agl[str(request["number"])]
        yaw, elevation = (0., -90.) if self.pose == "unknown" else optical_pose(t, bool(request["isEO"]), self.camera)
        tracker=getattr(self,'tracker',None)
        if tracker is not None:
            block=None
            if tracker.p['window_mode']=='blocks':
                # Block size of THIS frame from a small DSM read at the prior, so the block budget holds from frame 1.
                block=self.probe_block(request, t, camera, K, yaw, elevation, w, h)
                if block is None: block=getattr(self,'last_block_m',None)
            x,y=self.full_ref.forward.transform(t["lon"],t["lat"],errcheck=True)
            state=tracker.predict(int(request["number"]),x,y,t["altitude"],block_m=block)
            t["lon"],t["lat"]=self.full_ref.inverse.transform(state["x"],state["y"],errcheck=True)
            t["altitude"]=state["altitude"]
            self.full_ref.window_m=state["window_m"]
            self.trace['prior']={k:state[k] for k in ('mode','radius_m','window_m','window_cap_m','frames')}
            print('[PRIOR_TRACK] frame=%s mode=%s frames=%d radius=%.0fm window=%.0fm cap=%.0fm block=%s shift=%.0fm alt=%.1f'%(
                request["number"],state["mode"],state["frames"],state["radius_m"],state["window_m"],state["window_cap_m"],
                'unknown' if block is None else '%.0fm'%block,math.hypot(state["x"]-x,state["y"]-y),state["altitude"]),flush=True)
        oblique = self.oblique()
        agl0 = None; tv = t
        if oblique:
            # Slant camera looks km ahead: aircraft AGL from a small DSM read, then centre the map on the
            # optical-axis ground point instead of the aircraft (same as training scc_inference.frame_retrieval).
            agl0 = self.aircraft_agl(t, camera)
            x0, y0 = self.full_ref.forward.transform(t["lon"], t["lat"], errcheck=True)
            vx0, vy0 = view_centre(x0, y0, agl0, yaw, elevation)
            tv = dict(t); tv["lon"], tv["lat"] = self.full_ref.inverse.transform(vx0, vy0, errcheck=True)
        self.ref = self.full_ref.for_frame(tv)
        # C++ interface is RGB. Both deployed pipelines take BGR arrays.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std()) < 1.0:
            raise ValueError("query image has no usable intensity variation")
        dist = np.asarray(self.camera.get("distortion", [0, 0, 0, 0, 0]), dtype=float)
        if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
            raise ValueError("invalid distortion coefficients")
        if np.any(dist):
            image = cv2.undistort(image, K, dist)
        agl = agl0 if oblique else self.ref.agl(t, camera)
        need = self.window_for_block(request, K, agl, yaw, elevation, w, h)
        if need is not None and self.full_ref.window_m > 0 and need > self.full_ref.window_m:
            # A small (tracking) window cannot hold one rotated block -> REF_WINDOW failure. Enlarge once.
            self.full_ref.window_m = need
            self.ref = self.full_ref.for_frame(tv)
            agl = agl0 if oblique else self.ref.agl(t, camera)
            if isinstance(self.trace.get('prior'), dict): self.trace['prior']['window_m'] = need
            print('[PRIOR_TRACK] window enlarged to %.0fm for the block size' % need, flush=True)
        return t, image, K, agl, yaw, elevation

    def oblique(self):
        """Slant pipeline (camera.oblique): view-centred map window + bird's-eye query + masked tiles."""
        return bool(self.camera.get("oblique", False)) and getattr(self, "pose", "yp") == "yp"

    def aircraft_agl(self, t, camera):
        """AGL at the aircraft prior from a ~200 m DSM read (the search map is not centred there)."""
        old = self.full_ref.window_m
        try:
            self.full_ref.window_m = 200.
            return self.full_ref.for_frame(t).agl(t, camera)
        finally:
            self.full_ref.window_m = old

    def block_side(self, request, K, agl, yaw, elevation, w, h):
        """SCC block side (m) exactly as utils.retrieval_all estimates it; None if unavailable."""
        api = getattr(self, 'api', None)
        if api is None or not hasattr(api, 'estimate_reference_side'): return None
        data = dict(rel_height=agl, yaw=yaw, pitch=elevation, focal_len=float(K[0, 0]), focal_y=float(K[1, 1]),
                    roll_deg=float(request['telemetry'][11]) if bool(request["isEO"]) else 0.0,
                    reference_window_scale=float(self.camera.get("reference_window_scale", 1.2)),
                    is_slant=bool(request.get("isSlant", False)), pose_known=(self.pose == "yp"))
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                s = int(min(w, h) * float(self.camera.get("query_crop_scale", 1.0)))
                return float(api.estimate_reference_side(data, [s, s]))
        except Exception:
            return None

    def probe_block(self, request, t, camera, K, yaw, elevation, w, h):
        """Block side at the raw prior from a ~200 m DSM read, before the search window is chosen."""
        if getattr(self, 'api', None) is None or not getattr(self.full_ref, 'window_m', 0) > 0: return None
        old = self.full_ref.window_m
        try:
            self.full_ref.window_m = 200.
            return self.block_side(request, K, self.full_ref.for_frame(t).agl(t, camera), yaw, elevation, w, h)
        except Exception:
            return None
        finally:
            self.full_ref.window_m = old

    def window_for_block(self, request, K, agl, yaw, elevation, w, h):
        """Smallest search window that fits the rotated SCC block and >=2 block widths (None if unknown)."""
        side = self.block_side(request, K, agl, yaw, elevation, w, h)
        if side is None: return None
        self.last_block_m = side   # fallback for the next frame's window cap
        a = math.radians(yaw if self.pose == "yp" else 0.0)
        return float(max(1.1*side*(abs(math.cos(a)) + abs(math.sin(a))), 2.0*side))

    def confidence(self, inliers, reprojection=None):
        # Default is deliberately uncalibrated/0; success of PnP alone is not an accuracy guarantee.
        policy = self.section.get("self_eval", {})
        if not policy.get("enabled", False):
            return 0
        threshold = int(policy["min_inliers"])
        if threshold < 5:
            raise ValueError("self_eval.min_inliers must be >=5")
        passed = inliers >= threshold
        if "max_reprojection_px" in policy:
            passed = passed and reprojection is not None and np.isfinite(reprojection) and reprojection <= float(policy["max_reprojection_px"])
        return int(passed)


class AnyVisBackend(BaseBackend):
    def __init__(self, config, modality=1):
        super().__init__(config, modality)
        import avl_utils as api
        self.api = api
        self.settings["RETRIEVAL_METHODS"] = ["CAMP"]
        self.opt = SimpleNamespace(device=self.device, pose_priori=self.pose, strategy="Topn_opt", PnP_method="P3P",
            resize_ratio=self.tuning['resize_ratio'], patch_scale=self.tuning['anyvis_patch_scale'], min_patch_size_m=0.,
            max_patch_size_m=0., visualize=False, disable_ref_feature_cache=True,
            pnp_reproj_error=8., pnp_iterations=2000, pnp_confidence=.999)
        self.methods = api.retrieval_init({"retrieval_method": "CAMP"}, self.settings)
        self.methods["matching_method"] = "Roma"
        self.methods = api.matching_init(self.methods, self.opt, self.settings)
        configure_matcher(self,self.methods['matching_model'])
        api.cv2=CVProxy(self)
        def tuned_match(a,b,model,save_path,ransac_name,need_ransac=False,show_matches=False,device=None):
            left,right,_=match_pair(self,a,b,model,need_ransac,convert_rgb=False)
            return left,right
        api.Roma_match=tuned_match
        estimate=api.estimate_drone_pose_anyvisloc
        def timed_pose(*args,**kwargs):
            start=time.perf_counter()
            try:return estimate(*args,**kwargs)
            finally:self.trace['pose_s']+=time.perf_counter()-start
        api.estimate_drone_pose_anyvisloc=timed_pose
        # Upstream squeeze() makes a singleton gallery score scalar. Duplicate that one
        # candidate only; its geometry and feature values are unchanged.
        original = api.compute_block_mid_wo_black
        def gallery(*args, **kwargs):
            mids = original(*args, **kwargs)
            return np.repeat(mids, 2, axis=0) if len(mids) == 1 else mids
        api.compute_block_mid_wo_black = gallery
        self.torch.cuda.synchronize()

    def localize(self, request, image):
        t, image, K, agl, yaw, elevation = self.prepare(request, image)
        local_x, local_y = self.ref.local_prior(t)
        data = dict(x=local_x, y=local_y, z=t["altitude"]-self.ref.z0, rel_alt=agl,
                    yaw=yaw, pitch=90+elevation, roll=0, width=image.shape[1], height=image.shape[0], K=K)
        if self.pose == "yp":
            ref_image, rotation = self.api.dumpRotateImage(self.ref.image, yaw)
        else:
            ref_image, rotation = self.ref.image, np.eye(2, 3, dtype=np.float32)
        res = np.array([self.ref.res, self.ref.res], dtype=np.float32)
        origin = np.zeros(2, dtype=np.float32)
        self.trace['prepare_s']=time.perf_counter()-self.frame_start
        with tempfile.TemporaryDirectory(prefix="nav-rgb-") as run, self.torch.inference_mode():
            self.torch.cuda.synchronize();start=time.perf_counter()
            _, rows, cols, _, ph, pw, scale, _ = self.api.retrieval_all_anyvisloc(
                ref_image, image, data, res, origin, rotation, run, self.opt, self.settings, self.methods)
            self.torch.cuda.synchronize()
            self.trace['retrieval_s']=time.perf_counter()-start
            xyz,counts=[],[]
            seen_tiles=set()
            for rank in range(min(len(rows),self.tuning['max_candidates'])):
                tile=(rows[rank],cols[rank])
                if tile in seen_tiles:continue
                seen_tiles.add(tile)
                if self.trace['candidates_attempted']>=self.tuning['initial_candidates']:self.trace['fallback_used']=True
                self.last_geometry={};self.trace['candidates_attempted']+=1
                start=time.perf_counter()
                positions,n,_,_=self.api.Match2Pos_all_anyvisloc(
                    self.opt,self.settings,image,scale,K,ref_image,self.ref.dsm-self.ref.z0,
                    [rows[rank]],[cols[rank]],ph,pw,run,self.methods,rotation,res,origin,res,origin,
                    dist=np.zeros(5),truePos=None)
                self.torch.cuda.synchronize();self.trace['candidate_pipeline_s']+=time.perf_counter()-start
                xyz.extend(positions);counts.extend(n)
                point=positions[0] if positions else {}
                xy=[point['X'],point['Y']] if point.get('X') is not None and point.get('Y') is not None else None
                if self.candidate_finished(xy,n[0] if n else 0):break
        candidates = [i for i, p in enumerate(xyz) if counts[i] >= self.tuning['min_result_inliers'] and
                      all(p.get(k) is not None and np.isfinite(p[k]) for k in ("X", "Y", "Z"))]
        if not candidates:
            raise RuntimeError("AnyVisLoc: no valid PnP solution")
        best = max(candidates, key=lambda i: counts[i])
        if self.tuning['reject_weak_results'] and not good_enough(self.trace['candidates'],self.tuning):
            raise RuntimeError('AnyVisLoc: geometry quality gate not met')
        lon, lat, alt = self.ref.local_to_geographic(xyz[best])
        self.last_fix = (lon, lat, counts[best])
        return lon, lat, alt, self.confidence(counts[best])


class SCCBackend(BaseBackend):
    def __init__(self, config, modality=0):
        super().__init__(config, modality)
        import utils as api
        self.api = api
        self.region = "adapter"
        self.settings.update(RETRIEVAL_METHODS=["MINIMA_Roma_DINOv2"], adapter_HIGH_REF_initialX=self.ref.x0,
            adapter_HIGH_REF_initialY=self.ref.y0, adapter_HIGH_REF_resolution=self.ref.res,
            adapter_HIGH_DSM_resolution=self.ref.res, adapter_HIGH_DSM_COORDINATE=[0, 0],
            adapter_HIGH_REF_COORDINATE=[0, 0])
        # Preserve SCC's candidate filtering, GA-PnP and fusion; replace only its
        # hardcoded northern-UTM conversion with the declared map CRS.
        api.deg2utm = lambda region, config, lon, lat: self.ref.forward.transform(lon, lat, errcheck=True)
        api.utm2deg = lambda region, config, x, y: self.ref.inverse.transform(x, y, errcheck=True)
        estimate = api.estimate_drone_pose
        def estimate_valid(region, config, points, K, x, y, dsm, **kwargs):
            points=np.asarray(points)
            valid=np.isfinite(points).all(axis=1)&np.isfinite(x)&np.isfinite(y)&np.isfinite(dsm)
            start=time.perf_counter()
            try:return estimate(region, config, points[valid], K, np.asarray(x)[valid],
                                np.asarray(y)[valid], np.asarray(dsm)[valid], **kwargs)
            finally:self.trace['pose_s']+=time.perf_counter()-start
        api.estimate_drone_pose = estimate_valid
        api.cv2=CVProxy(self)
        solver=api.least_squares
        def bounded_solver(*args,**kwargs):
            if self.tuning['refine_max_nfev'] is not None:kwargs['max_nfev']=self.tuning['refine_max_nfev']
            return solver(*args,**kwargs)
        api.least_squares=bounded_solver
        self.opt = SimpleNamespace(device=self.device, Ref_type="HIGH", pose_priori=self.pose,
            resize_ratio=self.tuning['resize_ratio'], crop_gain=self.tuning['scc_crop_gain'])
        self.methods = api.matching_init({"matching_method": "MINIMA_Roma"})
        configure_matcher(self,self.methods['matching_model'].model)
        def tuned_match(a,b,wrapper,save_path,ransac_name,need_ransac=False,show_matches=False):
            return match_pair(self,a,b,wrapper.model,need_ransac,convert_rgb=True)
        api.MINIMA_match=tuned_match
        self.methods["retrieval_method"] = "MINIMA_Roma_DINOv2"
        backbone = self.methods["matching_model"].model.encoder.dinov2_vitl14
        if isinstance(backbone, (tuple, list)):
            backbone = backbone[0]
        self.methods = api.retrieval_init(self.methods, self.settings, DINOv2_shared=backbone)
        head = self.section.get('retrieval_head')
        if head:   # optional trained MLP retrieval head; without it retrieval is the original GeM-1024
            from retrieval_head import HeadApplier
            self.methods['retrieval_head'] = HeadApplier(head, self.device)
            print('[RETRIEVAL_HEAD] loaded %s (%d-d)' % (head, self.methods['retrieval_head'].output_dim), flush=True)
        self.torch.cuda.synchronize()

    def localize(self, request, image):
        t, image, K, agl, yaw, elevation = self.prepare(request, image)
        h, w = image.shape[:2]
        self.settings.update(adapter_HIGH_REF_initialX=self.ref.x0,adapter_HIGH_REF_initialY=self.ref.y0)
        data = dict(lon=t["lon"], lat=t["lat"], rel_height=agl, yaw=yaw, pitch=elevation,
                    width=w, height=h, cam_size=math.hypot(w, h), focal_len=float(K[0, 0]))
        data.update(
            focal_y=float(K[1, 1]),
            query_crop_scale=float(self.camera.get("query_crop_scale", 1.0)),
            roll_deg=float(t["inns_roll"]) if bool(request["isEO"]) else 0.0,
            reference_window_scale=float(self.camera.get("reference_window_scale", 1.2)),
            is_slant=bool(request.get("isSlant", False)),
            pose_known=(self.pose == "yp"),
            map_hw=tuple(self.ref.image.shape[:2]))
        if self.pose == "yp":
            ref_image, rotation = self.api.dumpRotateImage(self.ref.image, yaw)
        else:
            ref_image, rotation = self.ref.image, np.eye(2, 3, dtype=np.float32)
        mh, mw = self.ref.image.shape[:2]
        res = self.ref.res
        sx = self.ref.x0 + (mw - 1) * res / 2
        sy = self.ref.y0 - (mh - 1) * res / 2
        slon, slat = self.ref.inverse.transform(sx, sy)
        px, py = self.ref.forward.transform(t["lon"], t["lat"])

        vx, vy = self.api.view_center(
            self.region, self.settings, data,
            self.ref.x0, self.ref.y0, res, rotation)
        ux, uy = cv2.invertAffineTransform(rotation) @ np.array([vx, vy, 1.0])
        gx = self.ref.x0 + ux * res
        gy = self.ref.y0 - uy * res
        glon, glat = self.ref.inverse.transform(gx, gy)
        inside = (-0.5 <= ux < mw - 0.5 and -0.5 <= uy < mh - 0.5)

        print(f'[SEARCH_CENTER] frame={request["number"]} '
              f'prior_lon_lat=({t["lon"]:.8f},{t["lat"]:.8f}) '
              f'map_center_lon_lat=({slon:.8f},{slat:.8f}) '
              f'map_W_H_m=({mw*res:.1f},{mh*res:.1f}) '
              f'map_center_from_prior_m={math.hypot(sx-px, sy-py):.1f}', flush=True)
        print(f'[VIEW_CENTER] lon_lat=({glon:.8f},{glat:.8f}) '
              f'from_prior_m={math.hypot(gx-px, gy-py):.1f} '
              f'from_map_center_m={math.hypot(gx-sx, gy-sy):.1f} '
              f'inside_map_bounds={bool(inside)}', flush=True)
        print(f'[CENTER_POSE] isEO={request["isEO"]} pose={self.pose} '
              f'AGL={agl:.2f} yaw_used={yaw:.3f} elevation_used={elevation:.3f} '
              f'inns_YPR=({t["inns_yaw"]:.3f},{t["inns_pitch"]:.3f},{t["inns_roll"]:.3f})', flush=True)
        self.trace['prepare_s']=time.perf_counter()-self.frame_start
        self.methods['tile_mask'] = None
        staged = image
        if self.oblique():
            # Bird's-eye query on the exact ground square of one SCC block, heading up like the rotated map.
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                side_m = self.api.estimate_reference_side(data, self.api.resolution_size(data, self.opt)[1])
            side_px = max(10, 2 * int(math.ceil(side_m / (2 * res))))
            gray, mask = rectify_query(image[..., 0], K, agl, elevation, side_px * res)
            staged = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            data['query_prepared'] = True
            self.methods['tile_mask'] = mask
            print('[OBLIQUE] frame=%s block=%.0fm visible=%.2f' % (request["number"], side_px * res, float(mask.mean())), flush=True)
        with tempfile.TemporaryDirectory(prefix="nav-ir-") as run, self.torch.inference_mode():
            query = str(Path(run)/"query.png")
            if not cv2.imwrite(query, staged):
                raise RuntimeError("failed to stage query for SCC retrieval")
            self.torch.cuda.synchronize();start=time.perf_counter()
            _, rows, cols, _, widths, heights, scale, _, scores = self.api.retrieval_all(
                ref_image, query, data, self.ref.res, rotation, run, self.opt, self.region, self.settings, self.methods)
            self.torch.cuda.synchronize()
            self.trace['retrieval_s']=time.perf_counter()-start
            inv = cv2.invertAffineTransform(rotation)
            for k in range(min(5, len(rows))):
                ux, uy = inv @ np.array([cols[k] + widths[k]/2 - .5, rows[k] + heights[k]/2 - .5, 1.0])
                clon, clat = self.ref.inverse.transform(self.ref.x0 + ux*res, self.ref.y0 - uy*res)
                print(f'[RETRIEVAL_TOP] frame={request["number"]} rank={k+1} lon_lat=({clon:.8f},{clat:.8f}) '
                      f'score={float(np.ravel(scores)[k]):.4f} from_prior_m={math.hypot(self.ref.x0+ux*res-px, self.ref.y0-uy*res-py):.1f}', flush=True)
            blh,counts,errors,retrieval,uncertainty,consistent=[],[],[],[],[],[]
            one=lambda v,i:[v[i]] if isinstance(v,(list,tuple,np.ndarray)) else v
            centres=[consistency.block_centre(rotation,rows[k],cols[k],one(heights,k)[0],one(widths,k)[0],self.ref.x0,self.ref.y0,res)
                     for k in range(len(rows))]
            checks=[]   # per candidate (ok, reason, block distance m, dz m), also feeds selfEval
            # Tracking: the window is small and centred on the prediction -> fewer candidates, stop at the first good fix.
            tr=getattr(self,'tracker',None)
            st=tr.last if tr is not None and tr.last is not None and tr.last['number']==int(request['number']) else None
            tracking=st is not None and st['mode']=='track'
            limit=self.tuning['max_candidates']
            if tracking and self.tuning.get('track_max_candidates',0):limit=min(limit,self.tuning['track_max_candidates'])
            self.trace['candidate_limit']=limit
            for rank in range(min(len(rows),limit)):
                if self.trace['candidates_attempted']>=self.tuning['initial_candidates']:self.trace['fallback_used']=True
                self.last_geometry={};self.trace['candidates_attempted']+=1
                start=time.perf_counter()
                values=self.api.Match2Pos_all(self.opt,self.region,self.settings,image,scale,K,ref_image,self.ref.dsm,
                    [cols[rank]],[rows[rank]],one(heights,rank),one(widths,rank),run,self.methods,rotation,
                    retrieval_scores=one(scores,rank),prior_pitch=None if self.pose=='unknown' else elevation)
                self.torch.cuda.synchronize();self.trace['candidate_pipeline_s']+=time.perf_counter()-start
                pp,nn,_,_,ee,rr,_,uu=values
                blh.extend(pp);counts.extend(nn);errors.extend(ee);retrieval.extend(rr);uncertainty.extend(uu)
                point=pp[0] if pp else {}
                xy=list(self.ref.forward.transform(point['L'],point['B'])) if point.get('L') is not None and point.get('B') is not None else None
                ok,why,dist,dz=True,'no_fix',float('nan'),float('nan')
                if xy is not None and point.get('H') is not None and np.isfinite(point['H']):
                    ok,why,dist=consistency.check_fix(xy[0],xy[1],point['H'],t['altitude'],centres[rank],one(widths,rank)[0]*res,
                                                      agl,yaw,elevation,self.check)
                    dz=point['H']-t['altitude']
                    print('[PNP_CHECK] frame=%s rank=%d inliers=%d %s dist=%.0fm dz=%.0fm'%(request["number"],rank+1,nn[0] if nn else 0,
                          why,dist,dz),flush=True)
                    ok=ok or not self.check['enabled']
                consistent.extend([ok]*len(pp)); checks.extend([(ok,why,dist,dz,rank)]*len(pp))
                if self.candidate_finished(xy,nn[0] if nn else 0,ee[0] if ee else None) and ok:break
                if (tracking and ok and xy is not None and nn and nn[0]>=self.tuning['min_result_inliers']
                        and math.hypot(xy[0]-st['x'],xy[1]-st['y'])<=st['radius_m']):
                    print('[TRACK_FAST] frame=%s stop after candidate %d (fix within the tracking radius)'%(request["number"],rank+1),flush=True)
                    break
        valid = [i for i, p in enumerate(blh) if counts[i] >= self.tuning['min_result_inliers'] and
                 all(p.get(k) is not None and np.isfinite(p[k]) for k in ("L", "B", "H")) and
                 (not self.check['enabled'] or consistent[i])]
        if not valid:
            fallback=None
            if self.check['enabled'] and self.check['retrieval_fallback'] and agl<=self.check['max_fallback_agl']:
                fallback=consistency.retrieval_position(centres,list(np.ravel(scores)),one(widths,0)[0]*res,agl,yaw,elevation,self.check)
            if fallback is None:
                raise RuntimeError("SCC-Loc: no valid PnP solution")
            lon,lat=self.ref.inverse.transform(*fallback)
            self.last_source='retrieval'
            print('[FALLBACK] frame=%s retrieval position (no consistent PnP fix)'%request["number"],flush=True)
            return lon, lat, t["altitude"], 0
        take = lambda values: [values[i] for i in valid]
        index = int(self.api.select_best_index_fusion(take(counts), take(errors), take(retrieval), take(uncertainty), take(blh)))
        best = valid[index]
        if self.tuning['reject_weak_results'] and not good_enough(self.trace['candidates'],self.tuning):
            raise RuntimeError('SCC-Loc: geometry quality gate not met')
        self.last_fix = (blh[best]["L"], blh[best]["B"], counts[best])
        self.trace['self_eval_features'] = self.fix_features(blh, counts, errors, valid, best, checks, centres, scores, widths, res, one)
        value, _ = self_eval.decide(self.trace['self_eval_features'], 'fix', None, self.self_eval_cfg())
        return blh[best]["L"], blh[best]["B"], blh[best]["H"], value

    def fix_features(self, blh, counts, errors, valid, best, checks, centres, scores, widths, res, one):
        """Reliability features of the chosen fix (logged in [NAV_TIMING]; thresholds in infrared.self_eval)."""
        ok, why, dist, dz, rank = checks[best]
        side = float(one(widths, rank)[0] * res)
        s = np.ravel(scores).astype(float)
        far = [s[k] for k in range(len(centres)) if math.hypot(centres[k][0]-centres[rank][0], centres[k][1]-centres[rank][1]) > 1.01*side]
        xy = lambda i: self.ref.forward.transform(blh[i]['L'], blh[i]['B'])
        bx, by = xy(best)
        agreeing = sum(1 for i in valid if math.hypot(xy(i)[0]-bx, xy(i)[1]-by) <= 50.)
        err = errors[best]
        return dict(inliers=int(counts[best]), reprojection=None if err is None else float(err), block_dist=float(dist), side_m=side,
                    dz=float(dz), rank=int(rank)+1, score=float(s[rank]) if rank < len(s) else None,
                    score_margin=float(s[rank]-max(far)) if far and rank < len(s) else 1.0, agreeing=int(agreeing))
