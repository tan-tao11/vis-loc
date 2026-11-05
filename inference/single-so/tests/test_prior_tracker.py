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

    def test_turning_track_uses_fix_velocity(self):
        rng = np.random.default_rng(3); tr = PriorTracker(); pos = np.zeros(2); head = 0.; inside = []; locked = False
        for n in range(200):
            head += 0.06 if (n//40) % 2 else 0.; pos = pos + 150*np.array([math.cos(head), math.sin(head)])  # turns, 150 m/frame
            s = tr.predict(n, *noisy(rng, pos), 1000.)
            ok = max(abs(pos[0]-s['x']), abs(pos[1]-s['y'])) + 150 < s['window_m']/2
            if locked: inside.append(ok)
            if ok and tr.update(n, *(pos + rng.normal(0, 10, 2)), 100): locked = True
        self.assertTrue(locked)
        self.assertGreater(np.mean(inside), 0.9)

    def test_track_rejects_far_fix_and_reports_prediction(self):
        rng = np.random.default_rng(4); tr = PriorTracker()
        for n in range(30): tr.predict(n, *noisy(rng, np.array([30.*n, 0.])), 1000.)
        self.assertTrue(tr.update(29, 870., 0., 500))
        s = tr.predict(30, *noisy(rng, np.array([900., 0.])), 1000.)
        self.assertFalse(tr.update(30, 900.+400., 0., 500))             # 400 m off the prediction: wrong match
        x, y, source = tr.output(1300., 0., False)
        self.assertEqual(source, 'predicted'); self.assertLess(math.hypot(x-900., y), 150.)
        self.assertEqual(tr.output(None, None, False)[2], 'predicted')

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
        # localize_tracked: first frame (acquisition) reports the raw result and may accept it.
        backend.localize = lambda request, image: (backend.prepare(request, image), setattr(backend, 'last_fix', (87., 42., 500)))[0] and (87., 42., 1234., 1)
        backend.tracker = PriorTracker({'max_window_m': 1500.}); backend.begin_frame(request)
        out = backend.localize_tracked(request, gray)
        self.assertEqual(out, (87., 42., 1234., 1)); self.assertTrue(backend.trace['prior_fix_accepted'])
        # Next frame: localization fails while tracking -> predicted position, confidence 0, no exception.
        def fail(request, image):
            backend.prepare(request, image); raise RuntimeError('SCC-Loc: no valid PnP solution')
        backend.localize = fail; request = dict(request, number=6); backend.begin_frame(request)
        out = backend.localize_tracked(request, gray)
        self.assertEqual(backend.trace['output_source'], 'predicted'); self.assertEqual(out[3], 0)
        self.assertAlmostEqual(out[0], 87., places=3); self.assertEqual(out[2], 1234.)
        # Without a tracker, failures propagate exactly as before.
        backend.tracker = None
        with self.assertRaisesRegex(RuntimeError, 'no valid PnP'): backend.localize_tracked(request, gray)

if __name__ == '__main__':
    unittest.main()
