import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
import consistency
from oblique import view_offset


class ConsistencyTests(unittest.TestCase):
    cfg = consistency.settings({'enabled': True, 'retrieval_fallback': True})

    def test_nadir_fix_inside_outside_and_altitude(self):
        side = 370.
        self.assertTrue(consistency.check_fix(1000., 2000., 1500., 1500., (1050., 2000.), side, 1000., 30., -90., self.cfg)[0])
        ok, why, dist = consistency.check_fix(1300., 2000., 1500., 1500., (1000., 2000.), side, 1000., 30., -90., self.cfg)
        self.assertFalse(ok); self.assertEqual(why, 'outside_block'); self.assertAlmostEqual(dist, 300.)
        self.assertEqual(consistency.check_fix(1000., 2000., 1900., 1500., (1000., 2000.), side, 1000., 0., -90., self.cfg)[1], 'altitude')

    def test_oblique_fix_uses_look_ahead_with_its_own_height(self):
        side, agl, yaw, el = 3000., 900., 90., -15.   # heading east
        d = view_offset(agl + 100., el)               # fix 100 m higher than the prior
        centre = (5000. + d, 0.)
        self.assertTrue(consistency.check_fix(5000., 0., 2100., 2000., centre, side, agl, yaw, el, self.cfg)[0])
        self.assertFalse(consistency.check_fix(5000. - 3000., 0., 2100., 2000., centre, side, agl, yaw, el, self.cfg)[0])

    def test_retrieval_position(self):
        cfg = self.cfg
        x, y = consistency.retrieval_position([(0., 0.), (100., 0.), (5000., 0.)], [.80, .80, .70], 370., 1000., 0., -90., cfg)
        self.assertAlmostEqual(x, 50., places=6); self.assertAlmostEqual(y, 0.)
        self.assertIsNone(consistency.retrieval_position([(0., 0.), (5000., 0.)], [.80, .799], 370., 1000., 0., -90., cfg))
        x, y = consistency.retrieval_position([(0., 0.)], [.8], 3000., 900., 0., -15., cfg)     # heading north
        self.assertAlmostEqual(y, -view_offset(900., -15.), places=6); self.assertAlmostEqual(x, 0., places=6)

    def test_block_centre_identity_rotation(self):
        x, y = consistency.block_centre(np.eye(2, 3), 10, 20, 100, 100, 1000., 5000., 2.)
        self.assertAlmostEqual(x, 1000. + (20 + 50 - .5)*2.); self.assertAlmostEqual(y, 5000. - (10 + 50 - .5)*2.)

    def test_settings_default_off(self):
        self.assertFalse(consistency.settings(None)['enabled'])


class RetrievalFallbackOutputTests(unittest.TestCase):
    def test_retrieval_fallback_is_reported_but_never_starts_a_track(self):
        from backends import BaseBackend
        from prior_tracker import PriorTracker
        from pyproj import Transformer
        backend = BaseBackend.__new__(BaseBackend)
        class Ref:
            forward = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
            inverse = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True)
        backend.full_ref = Ref(); backend.tracker = PriorTracker({'max_window_m': 2000., 'offset_model': True})
        backend.tuning = {'profile_timing': False}; backend.config = {}; backend.mode = 0
        request = {'imgType': 0, 'number': 7}
        def localize(request, image):
            x, y = Ref.forward.transform(87., 42.)
            backend.tracker.predict(7, x, y, 1000.)
            backend.last_source = 'retrieval'                     # as SCCBackend.localize does on fallback
            return 87.001, 42.001, 1000., 0
        backend.localize = localize; backend.begin_frame(request)
        out = backend.localize_tracked(request, None)
        self.assertEqual(out, (87.001, 42.001, 1000., 0))
        self.assertEqual(backend.trace['output_source'], 'retrieval')
        self.assertIsNone(backend.tracker.fix)                    # tracker untouched by a non-PnP position


if __name__ == '__main__':
    unittest.main()


class SCCLocalizeCheckTests(unittest.TestCase):
    """SCCBackend.localize with a stub SCC api: inconsistent PnP fixes are dropped, retrieval fallback used."""
    def make(self, fixes, fallback=True):
        import contextlib, types
        from backends import SCCBackend
        from pyproj import Transformer
        b = SCCBackend.__new__(SCCBackend)
        fwd = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True); inv = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True)
        x0, y0 = fwd.transform(87., 42.)
        ref = types.SimpleNamespace(x0=x0 - 1000., y0=y0 + 1000., res=2., image=np.zeros((1000, 1000, 3), np.uint8),
                                    dsm=np.zeros((1000, 1000)), forward=fwd, inverse=inv, z0=0.)
        t = dict(lon=87., lat=42., altitude=1500., inns_roll=0., inns_yaw=0., inns_pitch=0.)
        b.prepare = lambda request, image: (t, np.zeros((64, 64, 3), np.uint8), np.eye(3), 1000., 0., -90.)
        b.ref = ref; b.full_ref = ref; b.settings = {}; b.region = 'r'; b.opt = None; b.methods = {}; b.camera = {}; b.pose = 'yp'
        b.config = {}; b.mode = 0; b.tracker = None
        b.tuning = dict(max_candidates=2, initial_candidates=2, min_result_inliers=5, reject_weak_results=False,
                        adaptive_candidates=False, profile_timing=False)
        b.check = consistency.settings({'enabled': True, 'retrieval_fallback': fallback})
        b.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None), inference_mode=contextlib.nullcontext)
        # two retrieval blocks of 185 px (370 m) whose centres are the aircraft position and 1 km east of it
        rows, cols = [500 - 92.5, 500 - 92.5], [500 - 92.5, 1000 - 92.5 - 0]
        calls = []
        def match(opt, region, settings, image, scale, K, ref_image, dsm, cl, rl, hl, wl, run, methods, rotation, **kw):
            calls.append(cl[0]); lon, lat, H, n = fixes[len(calls) - 1]
            return [dict(L=lon, B=lat, H=H)], [n], 0, 0, [1.], [.8], [None], [0.1]
        b.api = types.SimpleNamespace(dumpRotateImage=lambda img, yaw: (img, np.eye(2, 3)),
            view_center=lambda *a: (500., 500.),
            retrieval_all=lambda *a: ([0, 1], rows, cols, None, [185, 185], [185, 185], 1., 0., np.array([.81, .70])),
            Match2Pos_all=match, select_best_index_fusion=lambda *a: 0)
        b.confidence = lambda n, e: 0
        b.begin_frame({'number': 1})
        return b, (x0, y0)

    def test_inconsistent_fix_dropped_and_retrieval_used(self):
        from pyproj import Transformer
        inv = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True); fwd = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
        x0, y0 = fwd.transform(87., 42.)
        wrong = inv.transform(x0 + 2500., y0) + (1500., 200)        # 2.5 km off its block, many inliers
        b, _ = self.make([wrong, inv.transform(x0 + 3000., y0 - 2000.) + (1500., 200)])
        lon, lat, H, conf = b.localize({'number': 1, 'isEO': 0}, None)
        x, y = fwd.transform(lon, lat)
        self.assertLess(math.hypot(x - x0, y - y0), 40.)              # top block centre (score-weighted)
        self.assertEqual(b.last_source, 'retrieval'); self.assertIsNone(b.last_fix); self.assertEqual(conf, 0)

    def test_consistent_fix_kept(self):
        from pyproj import Transformer
        inv = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True); fwd = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
        x0, y0 = fwd.transform(87., 42.)
        good = inv.transform(x0 + 20., y0 - 10.) + (1490., 150)
        b, _ = self.make([good, good])
        lon, lat, H, conf = b.localize({'number': 1, 'isEO': 0}, None)
        self.assertAlmostEqual(lon, good[0]); self.assertIsNotNone(b.last_fix)

    def test_no_fallback_raises(self):
        from pyproj import Transformer
        inv = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True); fwd = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
        x0, y0 = fwd.transform(87., 42.)
        wrong = inv.transform(x0 + 2500., y0) + (1500., 200)
        b, _ = self.make([wrong, wrong], fallback=False)
        with self.assertRaises(RuntimeError): b.localize({'number': 1, 'isEO': 0}, None)


class ReportSuccessTests(unittest.TestCase):
    def test_unreported_source_raises_not_reported(self):
        from backends import BaseBackend, NotReported
        from prior_tracker import PriorTracker
        from pyproj import Transformer
        b = BaseBackend.__new__(BaseBackend)
        class Ref:
            forward = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
            inverse = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True)
        b.full_ref = Ref(); b.tuning = {'profile_timing': False}; b.config = {}; b.mode = 0
        b.section = {'report_success': ['fix', 'unconfirmed_fix', 'predicted']}
        def localize(request, image):
            x, y = Ref.forward.transform(87., 42.); b.tracker.predict(request['number'], x, y, 1000.)
            b.last_source = 'retrieval'; return 87.001, 42.001, 1000., 0
        b.localize = localize
        b.tracker = PriorTracker({'max_window_m': 2000., 'offset_model': True}); b.begin_frame({'number': 3})
        with self.assertRaises(NotReported): b.localize_tracked({'number': 3}, None)
        self.assertFalse(b.trace['reported']); self.assertEqual(b.trace['output_source'], 'retrieval')
        # no tracker: a PnP fix is reported, a retrieval fallback is not
        b.tracker = None; b.begin_frame({'number': 4})
        b.localize = lambda request, image: (setattr(b, 'last_fix', (87., 42., 300)), (87., 42., 1000., 1))[1]
        self.assertEqual(b.localize_tracked({'number': 4}, None), (87., 42., 1000., 1))
        b.begin_frame({'number': 5}); b.localize = lambda r, i: (setattr(b, 'last_source', 'retrieval'), (87., 42., 1000., 0))[1]
        with self.assertRaises(NotReported): b.localize_tracked({'number': 5}, None)


class SelfEvalTests(unittest.TestCase):
    good = dict(inliers=80, reprojection=1.5, block_dist=40., side_m=370., dz=20., score_margin=.05, agreeing=2)

    def test_decide_all_checks(self):
        import self_eval
        cfg = self_eval.settings({'enabled': True})
        self.assertEqual(self_eval.decide(self.good, 'fix', 30., cfg), (1, []))
        self.assertEqual(self_eval.decide(self.good, 'predicted', None, cfg)[0], 0)
        self.assertEqual(self_eval.decide(None, 'fix', None, cfg)[1], ['no_fix'])
        for key, val, reason in (('inliers', 10, 'inliers'), ('reprojection', 9., 'reprojection'), ('block_dist', 300., 'block'),
                                 ('dz', -400., 'altitude'), ('score_margin', -.02, 'retrieval'), ('agreeing', 0, 'agreement')):
            self.assertEqual(self_eval.decide(dict(self.good, **{key: val}), 'fix', None, cfg), (0, [reason]))
        self.assertEqual(self_eval.decide(self.good, 'fix', 250., cfg), (0, ['track']))
        self.assertEqual(self_eval.decide(self.good, 'fix', None, self_eval.settings(None)), (0, ['disabled']))

    def test_scc_localize_reports_features_and_self_eval(self):
        from pyproj import Transformer
        inv = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True); fwd = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
        x0, y0 = fwd.transform(87., 42.)
        good = inv.transform(x0 + 20., y0 - 10.) + (1490., 150)
        b, _ = SCCLocalizeCheckTests().make([good, good])
        b.section = {'self_eval': {'enabled': True, 'max_reprojection_px': 4}}
        lon, lat, H, value = b.localize({'number': 1, 'isEO': 0}, None)
        f = b.trace['self_eval_features']
        self.assertEqual(f['inliers'], 150); self.assertEqual(f['rank'], 1); self.assertEqual(f['agreeing'], 1)   # rank-2 block is 1 km away: its identical fix is inconsistent
        self.assertLess(f['block_dist'], 30.); self.assertAlmostEqual(f['dz'], -10.); self.assertAlmostEqual(f['score_margin'], .11)
        self.assertEqual(value, 1)


class TrackFastTests(unittest.TestCase):
    def test_tracking_matches_fewer_candidates(self):
        import types
        from pyproj import Transformer
        inv = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True); fwd = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
        x0, y0 = fwd.transform(87., 42.)
        good = inv.transform(x0 + 20., y0 - 10.) + (1490., 150)
        # not tracking: both candidates are matched
        b, _ = SCCLocalizeCheckTests().make([good, good]); b.localize({'number': 1, 'isEO': 0}, None)
        self.assertEqual(b.trace['candidates_attempted'], 2)
        # tracking: the first fix lies within the tracking radius -> stop after one candidate
        b, _ = SCCLocalizeCheckTests().make([good, good]); b.tuning['track_max_candidates'] = 2
        b.tracker = types.SimpleNamespace(last=dict(number=1, mode='track', x=x0, y=y0, radius_m=150.))
        b.localize({'number': 1, 'isEO': 0}, None)
        self.assertEqual(b.trace['candidates_attempted'], 1); self.assertEqual(b.trace['candidate_limit'], 2)
        # tracking with track_max_candidates=1 and a fix outside the radius: still only one candidate
        far = inv.transform(x0 + 400., y0) + (1490., 150)
        b, _ = SCCLocalizeCheckTests().make([far, good]); b.tuning['track_max_candidates'] = 1
        b.check = consistency.settings({'enabled': False})
        b.tracker = types.SimpleNamespace(last=dict(number=1, mode='track', x=x0, y=y0, radius_m=150.))
        b.localize({'number': 1, 'isEO': 0}, None)
        self.assertEqual(b.trace['candidates_attempted'], 1)
