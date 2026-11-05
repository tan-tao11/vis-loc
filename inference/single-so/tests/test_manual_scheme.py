import math
import unittest
import numpy as np
import ast
import contextlib
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'single-so/python'))
import backends
UTILS = ast.parse((ROOT/'projects/SCC-Loc/utils.py').read_text())
ns = dict(math=math, np=np, time=time, sqrt=math.sqrt)
for node in UTILS.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('estimate_reference_side', 'resolution_size', 'retrieval_all', 'view_center'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_scc_utils', 'exec'), ns)
estimate_reference_side = ns['estimate_reference_side']



def inputs(alpha=90, roll=0):
    f=512/(2*math.tan(math.radians(22.7902039/2)))
    return dict(rel_height=700, focal_len=f, focal_y=f, pitch=-alpha,
                roll_deg=roll, reference_window_scale=1.2,
                pose_known=True, is_slant=alpha<80)


class WindowTests(unittest.TestCase):
    def test_nadir(self):
        d=inputs()
        self.assertAlmostEqual(estimate_reference_side(d,[512,512]),1.2*700*512/d['focal_len'])

    def test_scale_and_height_linear(self):
        d=inputs(30)
        first=estimate_reference_side(d,[512,512])
        d['reference_window_scale']*=2
        self.assertAlmostEqual(estimate_reference_side(d,[512,512]),first*2)
        d['rel_height']*=2
        self.assertAlmostEqual(estimate_reference_side(d,[512,512]),first*4)

    def test_slant_increases_size(self):
        sizes=[estimate_reference_side(inputs(a),[512,512]) for a in [90,60,30,15.3483686]]
        self.assertTrue(all(a<b for a,b in zip(sizes,sizes[1:])))
        self.assertAlmostEqual(sizes[-1],10488.02322067291,places=6)

    def test_bounds_ray_footprint_with_roll(self):
        for a in [90,60,30,20]:
            for r in [0,1,10,30]:
                d=inputs(a,r)
                side=estimate_reference_side(d,[512,512])
                p,r=np.deg2rad([-a,r])
                Ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
                Rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
                B=np.array([[0,0,1],[1,0,0],[0,1,0]])
                t=256/d['focal_len']
                rays=np.array([[-t,-t,1],[t,-t,1],[t,t,1],[-t,t,1]])@(Ry@Rx@B).T
                xy=700*rays[:,:2]/rays[:,2:3]
                self.assertGreaterEqual(side+1e-7,np.ptp(xy,axis=0).max()*1.2)

    def test_invalid_horizon_or_unknown_slant(self):
        with self.assertRaisesRegex(ValueError,'horizon'):
            estimate_reference_side(inputs(12),[512,512])
        d=inputs(30);d['pose_known']=False
        with self.assertRaisesRegex(ValueError,'pose_priori'):
            estimate_reference_side(d,[512,512])
        d=inputs();d['rel_height']=-1
        with self.assertRaises(ValueError):estimate_reference_side(d,[512,512])




class ManualIntegrationTests(unittest.TestCase):
    def test_matching_debug_preserves_inputs(self):
        import tempfile
        import json
        import tuning
        with tempfile.TemporaryDirectory() as directory:
            owner=SimpleNamespace(config={'_config_dir':directory},mode=0,
                                  trace={'number':7,'candidates_attempted':2})
            a=np.full((12,15,3),[10,20,30],dtype=np.uint8)
            b=np.full((16,20,3),[40,50,60],dtype=np.uint8)
            left=np.array([[2.,3.],[4.,5.]])
            right=np.array([[3.,4.],[5.,6.]])
            confidence=np.array([.9,.8])
            tuning._debug_matching(owner,a,b,left,right,confidence,'before_ransac')
            folder=Path(directory)/'debug_matching/mode0_frame7_candidate2'
            saved=np.load(next(folder.glob('*_points.npz')))
            np.testing.assert_array_equal(saved['left'],left)
            np.testing.assert_array_equal(saved['right'],right)
            np.testing.assert_array_equal(np.array(Image.open(next(folder.glob('*_a.png')))),a)
            self.assertEqual(json.loads(next(folder.glob('*_stats.json')).read_text())['count'],2)

    def test_nadir_alignment_and_slant_guard(self):
        t=dict(inns_yaw=216.2197266,inns_pitch=0.,inns_roll=0.,
               sensor_yaw=0.,sensor_pitch=0.,sensor_roll=0.)
        self.assertEqual(backends.optical_pose(t,False,{'nadir_reference_alignment':True}), (216.2197266,-90.))
        t.update(inns_yaw=306.528717, sensor_yaw=.1977569, sensor_pitch=-.6976531, sensor_roll=.7965331)
        self.assertEqual(backends.optical_pose(t,False,{'nadir_reference_alignment':True}), (306.528717,-90.))
        t.update(inns_yaw=27.6749821,inns_pitch=-15.3483686,inns_roll=-.3845332)
        np.testing.assert_allclose(backends.optical_pose(t,True,{}), [27.6749821,-15.3483686])
        with self.assertRaises(ValueError):backends.optical_pose(t,True,{'nadir_reference_alignment':True})
        t.update(inns_pitch=-90,inns_roll=4,inns_yaw=20)
        self.assertEqual(backends.optical_pose(t,True,{'nadir_reference_alignment':True}),(24.,-90.))

    def test_actual_retrieval_square_and_scale(self):
        class Stop(Exception):pass
        env=dict(ns); captured={}
        # Recompile so the function globals point to the instrumented environment.
        node=next(n for n in UTILS.body if isinstance(n,ast.FunctionDef) and n.name=='retrieval_all')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'retrieval', 'exec'),env)
        def extract(*args):
            captured.update(block=args[4],finescale=sys._getframe(1).f_locals['finescale'])
            raise Stop()
        env.update(extract_features=extract, view_center=lambda *a:(500,500),
                   compute_block_mid_wo_black=lambda *a:np.array([[500,500]]),
                   crop_center=lambda *a:Image.new('RGB',(512,512)))
        d=dict(width=640,height=512,cam_size=math.hypot(640,512),focal_len=1000,focal_y=1000,
               rel_height=700,pitch=-90,yaw=0,reference_window_scale=1.2,pose_known=True,is_slant=False,map_hw=(1000,1000))
        opt=SimpleNamespace(Ref_type='HIGH',pose_priori='yp',crop_gain=3)
        cfg=dict(adapter_HIGH_REF_initialX=0,adapter_HIGH_REF_initialY=1000)
        methods=dict(retrieval_method='MINIMA_Roma_DINOv2',retrieval_cover=50,retrieval_model=None,img_transform=lambda a:a)
        args=[np.ones((1000,1000,3),np.uint8),'unused',d,1,np.eye(2,3),'',opt,'adapter',cfg,methods]
        with self.assertRaises(Stop):env['retrieval_all'](*args)
        self.assertEqual(captured['block'],[432,432])
        self.assertAlmostEqual(captured['finescale'],.7)
        d['map_hw']=(100,100)
        with self.assertRaisesRegex(ValueError,'REF_WINDOW'):env['retrieval_all'](*args)

    def test_real_center_prints_inverse_rotation(self):
        tree=ast.parse((ROOT/'single-so/python/backends.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SCCBackend')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='localize')
        start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                   and [a.id for a in n.targets[0].elts]==['mh','mw'])
        end=next(i for i in range(start,len(fn.body)) if isinstance(fn.body[i],ast.Assign)
                 and isinstance(fn.body[i].targets[0],ast.Subscript))
        code=compile(ast.Module(body=fn.body[start:end],type_ignores=[]),'actual_center_logs','exec')
        fwd=Transformer.from_crs(4326,32645,always_xy=True); inv=Transformer.from_crs(32645,4326,always_xy=True)
        lon,lat=87.17,42.1;px,py=fwd.transform(lon,lat)
        ref=SimpleNamespace(image=np.zeros((1000,1000,3),np.uint8),res=1.,x0=px-500,y0=py+500,forward=fwd,inverse=inv)
        env=dict(ns); env['deg2utm']=lambda region,cfg,lon,lat:fwd.transform(lon,lat)
        node=next(n for n in UTILS.body if isinstance(n,ast.FunctionDef) and n.name=='view_center')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'view_center','exec'),env)
        for yaw,pitch,inside in [(0,-90,True),(216,-90,True),(27.675,-15.348,False),(90,-60,True)]:
            owner=SimpleNamespace(ref=ref,api=SimpleNamespace(view_center=env['view_center']),region='adapter',settings={},pose='yp')
            rotation=cv2.getRotationMatrix2D((500,500),yaw,1);rotation[:,2]+=[123,456]
            t=dict(lon=lon,lat=lat,inns_yaw=yaw,inns_pitch=pitch,inns_roll=0)
            d=dict(lon=lon,lat=lat,rel_height=700,yaw=yaw,pitch=pitch)
            scope=dict(self=owner,np=np,cv2=cv2,math=math,t=t,data=d,rotation=rotation,
                       request=dict(number=1,isEO=1),agl=700,yaw=yaw,elevation=pitch)
            with contextlib.redirect_stdout(io.StringIO()) as output:exec(code,scope)
            self.assertIn('[VIEW_CENTER]',output.getvalue())
            self.assertEqual(scope['inside'],inside)
            self.assertAlmostEqual(math.hypot(scope['gx']-px,scope['gy']-py),700/math.tan(math.radians(-pitch)),delta=1.5)

if __name__=='__main__':unittest.main()
