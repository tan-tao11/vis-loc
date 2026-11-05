import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from prior_tracker import PriorTracker, scan_offsets


def noisy(rng, pos, lo=3704., hi=5556.):
    r, a = rng.uniform(lo, hi), rng.uniform(0, 2*math.pi)
    return pos + r*np.array([math.cos(a), math.sin(a)])


class PriorTrackerTests(unittest.TestCase):
    def test_smooth_acquisition_shrinks_independent_errors(self):
        rng = np.random.default_rng(1); tr = PriorTracker({'acquisition': 'smooth'}); errors = []
        for n in range(60):
            pos = np.array([30.*n, 0.]); s = tr.predict(n, *noisy(rng, pos), 1000+rng.uniform(-200, 200))
            errors.append(math.hypot(s['x']-pos[0], s['y']-pos[1]))
            self.assertEqual(s['mode'], 'smooth'); self.assertTrue(600. <= s['window_m'] <= 2000.)
        self.assertLess(np.mean(errors[40:]), 0.45*np.mean(errors[:3]))
        self.assertLess(abs(s['altitude']-1000.), 120.)

    def test_scan_tiles_cover_error_disk(self):
        tiles = scan_offsets(5600., 2000., 0.1)
        self.assertEqual(tiles[0], (0., 0.))
        for a in np.linspace(0, 2*math.pi, 73):
            for r in (0., 2000., 4000., 5600.):
                p = (r*math.cos(a), r*math.sin(a))
                self.assertTrue(any(max(abs(p[0]-dx), abs(p[1]-dy)) <= 1000. for dx, dy in tiles))

    def test_scan_follows_drifting_bias(self):
        """Continuous (drifting) prior error: smoothing alone never finds it, scanning does."""
        def acquire(mode):
            rng = np.random.default_rng(5); tr = PriorTracker({'acquisition': mode}); bias = np.array([4500., 2500.])
            for n in range(150):
                pos = np.array([100.*n, 0.]); bias = bias + rng.normal(0, 15, 2)
                s = tr.predict(n, *(pos+bias+rng.normal(0, 50, 2)), 1000.)
                if max(abs(pos[0]-s['x']), abs(pos[1]-s['y'])) + 150 < s['window_m']/2:
                    tr.update(n, *(pos+rng.normal(0, 10, 2)), 200)
                if tr.confirmed: return n
        self.assertIsNone(acquire('smooth'))
        self.assertIsNotNone(acquire('scan'))

    def test_tentative_needs_confirmation_and_track_gate(self):
        tr = PriorTracker({'acquisition': 'smooth'})
        s = tr.predict(0, 0., 0., 1000.)
        self.assertFalse(tr.update(0, 5., 0., 5))                         # too few inliers
        self.assertFalse(tr.update(0, s['window_m'], 0., 500))            # outside the searched window
        self.assertTrue(tr.update(0, 5., 0., 500)); self.assertFalse(tr.confirmed)
        s = tr.predict(1, 100., 0., 1000.)
        self.assertEqual(s['mode'], 'tentative'); self.assertLess(s['window_m'], 1000.)
        self.assertFalse(tr.update(1, 5.+400., 0., 500))                  # far from prediction: wrong match
        self.assertEqual(tr.output(405., 0., False)[2], 'predicted')
        self.assertTrue(tr.update(1, 40., 0., 500)); self.assertTrue(tr.confirmed)
        s = tr.predict(2, 0., 0., 1000.)
        self.assertEqual(s['mode'], 'track'); self.assertAlmostEqual(s['x'], 75., places=6)   # velocity from two fixes
        for n in range(3, 30): s = tr.predict(n, 0., 0., 1000.)
        self.assertEqual(s['mode'], 'smooth')                             # no fix for too long

    def test_turning_track_uses_fix_velocity(self):
        rng = np.random.default_rng(3); tr = PriorTracker(); pos = np.zeros(2); head = 0.; inside = []
        for n in range(300):
            head += 0.06 if (n//40) % 2 else 0.; pos = pos + 150*np.array([math.cos(head), math.sin(head)])
            s = tr.predict(n, *noisy(rng, pos), 1000.)
            ok = max(abs(pos[0]-s['x']), abs(pos[1]-s['y'])) + 150 < s['window_m']/2
            if tr.confirmed: inside.append(ok)
            if ok: tr.update(n, *(pos + rng.normal(0, 10, 2)), 100)
        self.assertGreater(len(inside), 50); self.assertGreater(np.mean(inside), 0.9)

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
        self.assertEqual(backend.trace['prior']['mode'], 'scan')       # default acquisition, centre tile first
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
