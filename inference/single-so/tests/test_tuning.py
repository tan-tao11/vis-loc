import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from tuning import LEGACY,validate_tuning,good_enough,configure_matcher,read_config,CVProxy


class TuningTests(unittest.TestCase):
    def test_bounds_and_unknown_keys(self):
        validate_tuning(LEGACY)
        for change in ({'coarse_resolution':421},{'refine_resolution':559},
                       {'initial_candidates':5,'max_candidates':2},{'sample_matches':0},
                       {'adaptive_candidates':'yes'},{'refine_max_nfev':1},{'typo':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_tuning(dict(LEGACY,**change))

    def test_agreement_guard(self):
        a=dict(inliers=80,reprojection=1,coverage=.3,xy=[0,0])
        self.assertFalse(good_enough([a],LEGACY))
        self.assertFalse(good_enough([a,dict(a,xy=[100,100])],LEGACY))
        self.assertFalse(good_enough([a,dict(a,coverage=.01)],LEGACY))
        self.assertTrue(good_enough([a,dict(a,xy=[2,3])],LEGACY))

    def test_real_model_attributes_change(self):
        model=SimpleNamespace();owner=SimpleNamespace(mode=0,tuning=dict(LEGACY,coarse_resolution=420,refine_resolution=560,sample_matches=1800))
        configure_matcher(owner,model)
        self.assertEqual((model.h_resized,model.w_resized,model.upsample_res),(420,420,(560,560)))
        self.assertTrue(model.upsample_preds)
        self.assertTrue(model.symmetric)

    def test_profile_and_dataset_single_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'navigation.yaml'
            p.write_text(yaml.safe_dump(dict(version=1,active_profile='fast',active_dataset='site',
                profiles={'fast':{'tuning':{'coarse_resolution':420,'refine_resolution':560,'sample_matches':1800}}},
                datasets={'site':{'enabled_modalities':[0],'infrared':{'project_root':'/existing'}}})))
            with patch.dict(os.environ,{},clear=True):
                c=read_config(p)
                self.assertEqual(c['tuning']['sample_matches'],1800)
                self.assertEqual(c['enabled_modalities'],[0])
            with patch.dict(os.environ,{'NAV_PROFILE':'typo'}),self.assertRaises(ValueError):read_config(p)

    def test_ransac_iterations_reach_opencv(self):
        p=dict(LEGACY,post_homography_iterations=1500,homography_reprojection_px=2.5)
        proxy=CVProxy(SimpleNamespace(tuning=p))
        with patch('tuning.cv2.findHomography',return_value=(None,None)) as f:
            proxy.findHomography([],[],8,3,maxIters=5000,confidence=.99)
            self.assertEqual(f.call_args.kwargs['maxIters'],1500)
            self.assertEqual(f.call_args.args[3],2.5)

if __name__=='__main__':unittest.main(verbosity=2)
