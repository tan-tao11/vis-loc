import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from prior_tracker import PriorTracker


def noisy(rng, pos, lo=3704., hi=5556.):
    r, a = rng.uniform(lo, hi), rng.uniform(0, 2*math.pi)
    return pos + r*np.array([math.cos(a), math.sin(a)])


class PriorTrackerTests(unittest.TestCase):
    def test_smoothing_shrinks_independent_errors(self):
        rng = np.random.default_rng(1); tr = PriorTracker(); errors = []
        for n in range(60):
            pos = np.array([30.*n, 0.]); p = noisy(rng, pos)
            s = tr.predict(n, *p, 1000+rng.uniform(-200, 200))
            errors.append(math.hypot(s['x']-pos[0], s['y']-pos[1]))
            self.assertLessEqual(s['window_m'], 2000.); self.assertGreaterEqual(s['window_m'], 600.)
        self.assertGreater(np.mean(errors[:3]), 3000.)
        self.assertLess(np.mean(errors[40:]), 0.45*np.mean(errors[:3]))   # ~2*sigma/sqrt(N) at the newest frame
        self.assertLess(abs(s['altitude']-1000.), 120.)

    def test_track_after_accepted_fix_and_gate(self):
        rng = np.random.default_rng(2); tr = PriorTracker()
        for n in range(30): tr.predict(n, *noisy(rng, np.array([30.*n, 0.])), 1000.)
        truth = np.array([30.*29, 0.])
        self.assertFalse(tr.update(29, *(truth+[20000, 0]), 500))      # far from smoothed prior: rejected
        self.assertFalse(tr.update(29, *truth, 5))                      # too few inliers: rejected
        self.assertTrue(tr.update(29, *truth, 500))
        s = tr.predict(30, *noisy(rng, np.array([900., 0.])), 1000.)
        self.assertEqual(s['mode'], 'track'); self.assertLess(math.hypot(s['x']-900., s['y']), 200.)
        self.assertLess(s['window_m'], 1000.)
        for n in range(31, 60): s = tr.predict(n, *noisy(rng, np.array([30.*n, 0.])), 1000.)
        self.assertEqual(s['mode'], 'smooth')                            # no fix for too long

    def test_new_sequence_resets(self):
        tr = PriorTracker()
        for n in range(10): tr.predict(100+n, 1e4, 0., 1000.)
        s = tr.predict(3, 0., 0., 1000.)
        self.assertEqual(s['frames'], 1); self.assertEqual(tr.fix, None)


class BackendIntegrationTests(unittest.TestCase):
    def test_prepare_uses_tracked_prior_and_window(self):
        from backends import BaseBackend
        from pyproj import Transformer
        backend = BaseBackend.__new__(BaseBackend)
        backend.mode = 0; backend.camera = {'agl_m': 100}; backend.pose = 'unknown'
        seen = {}
        class Ref:
            window_m = 600.
            forward = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
            inverse = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True)
            def agl(self, t, c): return c['agl_m']
            def for_frame(self, t): seen.update(t=dict(t), window=self.window_m); return self
        backend.ref = backend.full_ref = Ref(); backend.tracker = PriorTracker({'max_window_m': 1500.})
        gray = np.full((16, 16), 55, np.uint8); gray[-1] = 180
        request = {'imgType': 0, 'isEO': 1, 'number': 5, 'telemetry': [87., 42., 1000., 0, 0, 0, 0, 0, 0, 0, -80, 0, 50, 40, 1]}
        backend.tuning = {'profile_timing': False}; backend.config = {}
        backend.begin_frame(request); backend.prepare(request, gray)
        self.assertEqual(seen['window'], 1500.)                          # first frame: budget cap
        self.assertAlmostEqual(seen['t']['lon'], 87., places=6)
        self.assertEqual(backend.trace['prior']['mode'], 'smooth')
        backend.last_fix = (87., 42., 500)
        backend.end_frame(True)
        self.assertTrue(backend.trace['prior_fix_accepted'])


if __name__ == '__main__':
    unittest.main()
