import math
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from oblique import rectify_query, view_offset, view_centre


class ObliqueTests(unittest.TestCase):
    def test_view_offset_and_centre(self):
        self.assertAlmostEqual(view_offset(1000., -15.), 1000./math.tan(math.radians(15.)), places=6)
        self.assertEqual(view_offset(800., -90.), 0.)
        x, y = view_centre(0., 0., 1000., 90., -45.)        # heading east, 45 deg down -> 1 km east
        self.assertAlmostEqual(x, 1000., places=6); self.assertAlmostEqual(y, 0., places=6)
        with self.assertRaises(ValueError): view_offset(1000., 5.)

    def test_rectify_heading_up_and_mask(self):
        # Flat-ground image: the top image row is farther ahead -> must land in the upper output rows.
        w, h = 320, 256
        K = np.array([[w/(2*math.tan(math.radians(14))), 0, (w-1)/2], [0, h/(2*math.tan(math.radians(11))), (h-1)/2], [0, 0, 1.]])
        gray = np.repeat(np.linspace(250, 5, h).astype(np.uint8)[:, None], w, 1)   # bright = far (top)
        out, mask = rectify_query(gray, K, 900., -15., 3000.)
        self.assertEqual(out.shape, (336, 336)); self.assertTrue(set(np.unique(mask)) <= {0, 1})
        rows = np.where(mask.any(1))[0]
        top, bottom = out[rows[:15]][mask[rows[:15]] > 0], out[rows[-15:]][mask[rows[-15:]] > 0]
        self.assertGreater(top.mean(), bottom.mean() + 40)
        self.assertTrue((out[mask == 0] == 0).all())


class ObliquePrepareTests(unittest.TestCase):
    def test_map_window_centred_on_view_not_aircraft(self):
        from backends import BaseBackend
        from pyproj import Transformer
        backend = BaseBackend.__new__(BaseBackend)
        backend.mode = 0; backend.pose = 'yp'
        backend.camera = {'agl_m': 1000., 'oblique': True, 'nadir_reference_alignment': False}
        seen = []
        class Ref:
            window_m = 6000.
            forward = Transformer.from_crs('EPSG:4326', 'EPSG:32645', always_xy=True)
            inverse = Transformer.from_crs('EPSG:32645', 'EPSG:4326', always_xy=True)
            def agl(self, t, c): return c['agl_m']
            def for_frame(self, t): seen.append((dict(t), self.window_m)); return self
        backend.ref = backend.full_ref = Ref(); backend.tracker = None
        gray = np.full((16, 16), 55, np.uint8); gray[-1] = 180
        # inns yaw 0 (north), inns pitch -15 deg, EO slant
        request = {'imgType': 0, 'isEO': 1, 'isSlant': 1, 'number': 5,
                   'telemetry': [87., 42., 2000., 0, 0, 0, 0, 0, 0, 0., -15., 0, 28., 22., 12]}
        backend.tuning = {'profile_timing': False}; backend.config = {}
        backend.begin_frame(request); t, *_ = backend.prepare(request, gray)
        self.assertEqual(seen[0][1], 200.)                               # small AGL probe at the aircraft
        self.assertAlmostEqual(seen[0][0]['lat'], 42., places=9)
        x0, y0 = Ref.forward.transform(87., 42.); x1, y1 = Ref.forward.transform(seen[-1][0]['lon'], seen[-1][0]['lat'])
        self.assertAlmostEqual(math.hypot(x1-x0, y1-y0), 1000./math.tan(math.radians(15.)), delta=1.)
        self.assertGreater(y1 - y0, 3700.)                              # ahead = north
        self.assertEqual(seen[-1][1], 6000.)
        self.assertAlmostEqual(t['lat'], 42., places=9)                  # returned prior stays the aircraft


if __name__ == '__main__':
    unittest.main()
