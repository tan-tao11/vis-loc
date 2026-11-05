"""Search-centre prior for video sequences with large, independent per-frame prior errors.

Shared by inference (single-so/python/prior_tracker.py) and training (training/scc_prior.py is a
byte copy; a test compares them). Coordinates are working-CRS metres, frames are ordered by
`number`. No ground truth is used anywhere in this module.

* smooth: robust straight-line fit of the recent priors (x, y, altitude) against frame number,
  evaluated at the current frame. Independent errors shrink roughly with 1/sqrt(N).
* track: after an accepted fix, centre = last fix + velocity * frames elapsed, with a small radius
  that grows per frame. Velocity comes from the last two accepted fixes (follows turns); with only
  one fix it falls back to the smoothed slope. Falls back to smooth after too many frames without a fix.
  The smoothing itself is a straight line over `history` frames: turn bias is tens to ~200 m, small
  against the km-level prior noise (simulated straight / 3.4 deg-per-frame / U-turn / S-curve).
The search window side is 2*(radius + margin), clipped to [min_window_m, max_window_m];
max_window_m is the per-frame time budget knob.

Output (score: a frame with error > 150 m scores 0, frames are averaged): an accepted fix is reported;
while tracking, a missing or inconsistent fix is replaced by the predicted position (last fix +
velocity), which is usually within tens of metres; during acquisition the raw result (or the smoothed
prior when there is none) is reported. While tracking, a fix is accepted only within the track radius
of the prediction (150 m + 20 m per frame since the last fix), so a wrong match cannot drag the track.
"""
import math
import numpy as np

DEFAULTS = dict(
    enabled=True,
    prior_error_m=5600.,      # worst-case horizontal prior error (3 nmi)
    altitude_error_m=200.,    # worst-case altitude prior error
    history=30,               # frames used by the line fit
    confidence=2.5,           # radius = confidence * predicted standard error
    margin_m=300.,            # added to the radius: roughly half a footprint
    min_window_m=600.,
    max_window_m=2000.,       # time budget: blocks grow with window^2
    track_radius_m=150.,
    track_growth_m=20.,       # extra radius per frame since the last accepted fix
    max_track_gap=15,         # frames without accepted fix before falling back to smooth
    min_fix_inliers=40,
    reset_gap=100)            # frame-number jump (or decrease) that starts a new sequence


def _robust_line(n, v, iterations=3):
    """Weighted least-squares v = a + b*n with residual-based down-weighting; returns a, b, sigma."""
    w = np.ones_like(v)
    a, b, sigma = float(np.median(v)), 0., float('inf')
    for _ in range(iterations):
        if w.sum() < 2 or np.ptp(n[w > 0]) == 0:
            a = float(np.average(v, weights=np.maximum(w, 1e-9))); b = 0.
        else:
            A = np.stack([np.ones_like(n), n], 1) * np.sqrt(w)[:, None]
            a, b = np.linalg.lstsq(A, v*np.sqrt(w), rcond=None)[0]
        r = v - (a + b*n)
        sigma = 1.4826*float(np.median(np.abs(r))) if len(r) > 2 else float('inf')
        if not math.isfinite(sigma) or sigma <= 0: break
        w = (np.abs(r) <= 3*sigma).astype(float)
    return float(a), float(b), sigma


class PriorTracker:
    def __init__(self, cfg=None):
        self.p = dict(DEFAULTS, **(cfg or {}))
        self.reset()

    def reset(self):
        self.priors = []          # (number, x, y, altitude)
        self.fix = None           # (number, x, y)
        self.prev_fix = None
        self.last = None

    def _smooth(self, number):
        p = self.p; h = np.asarray(self.priors[-int(p['history']):], float)
        n = h[:, 0] - number; N = len(h)
        sx = sy = p['prior_error_m']/math.sqrt(2)    # error ring of radius R, random direction
        out = {}
        for k, col, prior_sigma in (('x', 1, sx), ('y', 2, sy), ('altitude', 3, p['altitude_error_m']/math.sqrt(3))):
            a, b, s = _robust_line(n, h[:, col])
            s = prior_sigma if not math.isfinite(s) else min(max(s, 1.), 3*prior_sigma)
            # Standard error of a line fit at its newest end ~ 2*sigma/sqrt(N); one frame -> sigma.
            se = s if N < 3 else s*math.sqrt(max(1., (4*N-2)/(N*(N+1))))
            out[k] = a; out[k+'_rate'] = b; out[k+'_se'] = min(se, prior_sigma)
        radius = p['confidence']*math.hypot(out['x_se'], out['y_se'])/math.sqrt(2)
        out['radius_m'] = min(radius, p['prior_error_m']); out['frames'] = N
        return out

    def predict(self, number, x, y, altitude):
        """Register this frame's prior; return the search centre, altitude, radius and window."""
        p = self.p
        if self.priors and (number <= self.priors[-1][0] or number - self.priors[-1][0] > p['reset_gap']):
            self.reset()
        self.priors.append((float(number), float(x), float(y), float(altitude)))
        s = self._smooth(number)
        state = dict(mode='smooth', x=s['x'], y=s['y'], altitude=s['altitude'], radius_m=s['radius_m'],
                     frames=s['frames'], smooth_x=s['x'], smooth_y=s['y'], smooth_radius_m=s['radius_m'])
        if self.fix is not None:
            gap = number - self.fix[0]
            radius = p['track_radius_m'] + p['track_growth_m']*gap
            if gap <= p['max_track_gap'] and radius < s['radius_m']:
                vx, vy = s['x_rate'], s['y_rate']
                if self.prev_fix is not None and 0 < self.fix[0]-self.prev_fix[0] <= p['max_track_gap']:
                    d = self.fix[0]-self.prev_fix[0]   # velocity from the last two fixes follows turns
                    vx, vy = (self.fix[1]-self.prev_fix[1])/d, (self.fix[2]-self.prev_fix[2])/d
                state.update(mode='track', x=self.fix[1] + vx*gap, y=self.fix[2] + vy*gap, radius_m=radius)
            else:
                self.fix = self.prev_fix = None
        side = 2*(state['radius_m'] + p['margin_m'])
        state['window_m'] = float(min(max(side, p['min_window_m']), p['max_window_m']))
        state['number'] = number
        self.last = state
        return state

    def update(self, number, x, y, inliers):
        """Offer a localization result; accepted only if strong and consistent with the smoothed prior."""
        s = self.last
        if s is None or s['number'] != number: return False
        if inliers < self.p['min_fix_inliers']: return False
        if math.hypot(x - s['smooth_x'], y - s['smooth_y']) > s['smooth_radius_m'] + self.p['margin_m']: return False
        if s['mode'] == 'track' and math.hypot(x - s['x'], y - s['y']) > s['radius_m']: return False
        self.prev_fix, self.fix = self.fix, (float(number), float(x), float(y))
        return True

    def output(self, x=None, y=None, accepted=False):
        """Position to report for the frame of the last predict(): (x, y, source)."""
        s = self.last
        if accepted: return x, y, 'fix'
        if s['mode'] == 'track': return s['x'], s['y'], 'predicted'
        if x is not None: return x, y, 'unconfirmed_fix'
        return s['smooth_x'], s['smooth_y'], 'smoothed_prior'
