import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from backends import Reference, camera_matrix, optical_pose, load_config, BaseBackend


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.raw=self.root/'image.raw'; self.raw.write_bytes(bytes([12,54,240])*16*16)
        self.cfg=self.root/'config.json'; self.cfg.write_text('{}')

    def tearDown(self): self.temp.cleanup()

    def probe(self, behavior='ok'):
        env=dict(os.environ,NAV_WORKER=str(ROOT/'tests/mock_worker.py'),NAV_PYTHON=sys.executable,
                 NAV_STEP_TIMEOUT_MS='150',NAV_INIT_TIMEOUT_MS='5000',TEST_BEHAVIOR=behavior)
        return subprocess.run([str(ROOT/'build/adapter_probe'),str(self.cfg),str(self.raw),'16','16','RITR'],
                              env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)

    def test_mixed_modality_and_lifecycle(self):
        p=self.probe(); self.assertEqual(p.returncode,0,p.stderr)
        rows=[r.split() for r in p.stdout.splitlines() if r.startswith('RESULT')]
        self.assertEqual([r[1] for r in rows],list('RITR'))
        self.assertEqual([float(r[3]) for r in rows],[111,110,110,111])
        self.assertEqual([float(r[5]) for r in rows],[101,102,103,104])

    def test_worker_failure_is_bounded(self):
        for behavior in ('crash','timeout','bad_result','fail','init_fail'):
            with self.subTest(behavior=behavior):
                p=self.probe(behavior)
                self.assertIn(p.returncode,[1,3],p.stderr)
                for row in p.stdout.splitlines():
                    if row.startswith('RESULT'): self.assertEqual(row.split()[2:],["0","0","0","0","0"])

    def test_fov_and_calibration_scaling(self):
        K=camera_matrix({},dict(hfov=90,vfov=90),640,480)
        np.testing.assert_allclose(K,[[320,0,319.5],[0,240,239.5],[0,0,1]],atol=1e-8)
        K=camera_matrix(dict(K=[[100,0,40],[0,120,30],[0,0,1]],image_size=[80,60]),{},160,120)
        np.testing.assert_allclose(K,[[200,0,80],[0,240,60],[0,0,1]])
        with self.assertRaises(ValueError): camera_matrix({},dict(hfov=0,vfov=90),640,480)

    def test_coordinate_and_height_round_trip(self):
        cv2.imwrite(str(self.root/'map.png'),np.full((32,32,3),100,np.uint8))
        np.save(self.root/'dsm.npy',np.full((32,32),100,np.float32))
        ref=Reference(dict(image=str(self.root/'map.png'),dsm=str(self.root/'dsm.npy'),origin_xy=[500000,3000000],
                           resolution_m=2,crs='EPSG:32650'))
        lon,lat,z=ref.local_to_geographic(dict(X=10,Y=20,Z=50))
        x,y=ref.local_prior(dict(lon=lon,lat=lat))
        np.testing.assert_allclose([x,y,z],[10,20,150],atol=1e-6)
        self.assertAlmostEqual(ref.agl(dict(lon=lon,lat=lat,altitude=z),{}),50)
        with self.assertRaises(ValueError): ref.agl(dict(lon=lon,lat=lat,altitude=90),{})

    def test_pose_requires_known_optical_axis(self):
        t=dict(inns_yaw=30,inns_pitch=-60,inns_roll=0)
        yaw,elev=optical_pose(t,True,{})
        np.testing.assert_allclose([yaw,elev],[30,-60],atol=1e-8)
        with self.assertRaises(ValueError): optical_pose(t,False,{})

    def test_window_coordinates_and_nodata(self):
        import tifffile
        image=np.zeros((96,96,3),np.uint8)
        image[:,:,0]=np.arange(96,dtype=np.uint8)[None,:]
        tifffile.imwrite(self.root/'map.tif',image,photometric='rgb')
        heights=np.full((96,96),80,np.float32);heights[47,47]=0
        tifffile.imwrite(self.root/'height.tif',heights,rowsperstrip=1)
        ref=Reference(dict(image=str(self.root/'map.tif'),dsm=str(self.root/'height.tif'),
            origin_xy=[500000,3000000],resolution_m=1,crs='EPSG:32650',window_m=40,nodata=0))
        self.assertIsNone(ref.image)
        lon,lat=ref.inverse.transform(500048,2999952)
        view=ref.for_frame(dict(lon=lon,lat=lat))
        self.assertEqual(view.image.shape,(40,40,3))
        self.assertAlmostEqual(view.x0,500028,delta=1)
        col,row=int(500047-view.x0),int(view.y0-2999953)
        self.assertTrue(np.isnan(view.dsm[row,col]))
        self.assertEqual(view.image[row,col].tolist(),[0,0,0])
        self.assertEqual(view.image[0,0,2],int(view.x0-500000))

    def test_rgb_bgr_and_gray(self):
        backend=BaseBackend.__new__(BaseBackend)
        backend.mode=1; backend.camera={'agl_m':10}; backend.pose='unknown'
        class Ref:
            def agl(self,t,c): return c['agl_m']
            def for_frame(self,t): return self
        backend.ref=Ref()
        backend.full_ref=backend.ref
        request={'imgType':1,'isEO':1,'number':1,'telemetry':[110,30,100,0,0,0,0,0,0,0,0,0,90,90,1]}
        img=np.tile(np.uint8([12,54,240]),(16,16,1))
        img[-1]=180
        _,bgr,*_=backend.prepare(request,img)
        self.assertEqual(bgr[0,0].tolist(),[240,54,12])
        backend.mode=0; request['imgType']=0
        gray=np.full((16,16),55,np.uint8); gray[-1]=180
        _,bgr,*_=backend.prepare(request,gray)
        self.assertEqual(bgr[0,0].tolist(),[55,55,55])
        with self.assertRaisesRegex(ValueError,'intensity variation'):
            backend.prepare(request,np.full((16,16),55,np.uint8))

    def test_missing_reference_is_explicit_failure(self):
        target=ROOT/'navigation.yaml'
        if target.exists():
            with patch.dict(os.environ,{'NAV_DATASET':'desert'}),self.assertRaisesRegex(ValueError,'paths must be filled'):
                load_config(target)
        else:
            with self.assertRaisesRegex(ValueError,'paths must be filled'):load_config(ROOT/'naviParams.txt')


if __name__=='__main__': unittest.main(verbosity=2)
