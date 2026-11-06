"""Search centre and window per frame for video sequences with large prior errors (inference).

Coordinates are working-CRS metres, frames are ordered by `number`. No ground truth is used.

Primary mechanism: fix-based tracking. Once the previous frames were localized consistently, the
centre is last fix + velocity (from the last two fixes) and the window is small. This does not
depend on how the prior error behaves. A first acquisition fix is only TENTATIVE; it becomes a
track after a second consistent fix in the small window around it, so one wrong match cannot start
a wrong track.

Acquisition (no confirmed track) works for both random and slowly drifting prior errors:
* scan (default): windows of the per-frame window cap tile the disk of radius prior_error_m around
  the smoothed prior, one tile per frame (gap-free square grid, nearest first).
* smooth: search only the smoothed centre; fastest for independent errors, fails for drifting ones.
* hybrid: alternate scan and smooth frames.

Window cap (time budget), two schemes selected by window_mode:
* meters (default): cap = max_window_m. The window size is predictable; the number of SCC blocks
  (~ (2*window/block - 1)^2, block ~ 0.37*AGL for the vertical camera) varies with AGL.
* blocks: cap = block*(sqrt(max_blocks)+1)/2, so the number of blocks (time per frame) stays roughly
  constant whatever the AGL; the block size of the current frame is probed from the DSM before the
  window is chosen. hard_cap_m bounds the map read if the block estimate is unavailable or wrong.

Simulated score with scan (100 m/frame, S-curves, 70% retrieval success; independent / drifting
errors): 2 km window 0.49 / 0.53, 3 km 0.67 / 0.75, 4 km 0.75 / 0.84.

Output (score: error > 150 m scores 0, frames averaged): an accepted fix is reported; while tracking
(or tentative), a failed/untrusted fix is replaced by the predicted position; during acquisition the
raw result (or the centre searched) is reported.
"""
import math
import numpy as np

DEFAULTS = dict(
    enabled=True,
    acquisition='scan',       # scan | smooth | hybrid
    prior_error_m=5600.,      # worst-case horizontal prior error (3 nmi)
    altitude_error_m=200.,    # worst-case altitude prior error
    history=30,               # frames used by the smoothing line fit
    confidence=2.5,           # smooth radius = confidence * predicted standard error
    margin_m=300.,            # added to the radius: roughly half a footprint
    min_window_m=600.,
    window_mode='meters',     # meters: cap = max_window_m | blocks: cap from max_blocks and the block size
    max_window_m=4000.,       # meters mode: window cap
    max_blocks=100,           # blocks mode: approximate SCC blocks per frame
    hard_cap_m=8000.,         # blocks mode: absolute window cap (also used when the block size is unknown)
    scan_overlap=0.1,         # neighbouring scan tiles overlap by this fraction
    track_radius_m=150.,
    track_growth_m=20.,       # extra radius per frame since the last accepted fix
    max_track_gap=15,         # frames without accepted fix before returning to acquisition
    max_tentative_gap=2,      # frames a tentative fix waits for confirmation
    min_fix_inliers=40,
    reset_gap=100,            # frame-number jump (or decrease) that starts a new sequence
    offset_model=False)       # True: prior error is a constant offset; centre = this prior + (last fix - its prior)


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


def scan_offsets(radius, window, overlap=0.1):
    """Tile centres (dx, dy) covering a disk of `radius`, nearest first.

    Square grid with spacing step = window*(1-overlap). Every grid cell (side `step`) that touches
    the disk is kept; each cell lies inside its window, so the disk is covered without gaps.
    """
    step = max(1., window*(1-overlap)); n = int(math.ceil(radius/step + 0.5)); out = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            x, y = i*step, j*step
            if math.hypot(max(abs(x)-step/2, 0.), max(abs(y)-step/2, 0.)) <= radius: out.append((x, y))
    out.sort(key=lambda p: (round(math.hypot(*p), 6), math.atan2(p[1], p[0])))
    return out


class PriorTracker:
    def __init__(self, cfg=None):
        self.p = dict(DEFAULTS, **(cfg or {}))
        if self.p['acquisition'] not in ('scan', 'smooth', 'hybrid'): raise ValueError('acquisition must be scan, smooth or hybrid')
        if self.p['window_mode'] not in ('meters', 'blocks'): raise ValueError('window_mode must be meters or blocks')
        self.reset()

    def reset(self):
        self.priors = []          # (number, x, y, altitude)
        self.fix = self.prev_fix = None   # (number, x, y)
        self.confirmed = False
        self.acquire_count = 0
        self.last = None

    def _smooth(self, number):
        p = self.p; h = np.asarray(self.priors[-int(p['history']):], float)
        n = h[:, 0] - number; N = len(h)
        sigma = p['prior_error_m']/math.sqrt(2)
        out = {}
        for k, col, prior_sigma in (('x', 1, sigma), ('y', 2, sigma), ('altitude', 3, p['altitude_error_m']/math.sqrt(3))):
            a, b, s = _robust_line(n, h[:, col])
            s = prior_sigma if not math.isfinite(s) else min(max(s, 1.), 3*prior_sigma)
            se = s if N < 3 else s*math.sqrt(max(1., (4*N-2)/(N*(N+1))))
            out[k] = a; out[k+'_rate'] = b; out[k+'_se'] = min(se, prior_sigma)
        radius = p['confidence']*math.hypot(out['x_se'], out['y_se'])/math.sqrt(2)
        out['radius_m'] = min(radius, p['prior_error_m']); out['frames'] = N
        return out

    def window_cap(self, block_m=None):
        """Largest window for this frame (see window_mode)."""
        p = self.p
        if p['window_mode'] == 'blocks':
            cap = float(p['hard_cap_m'])
            if block_m: cap = min(cap, float(block_m)*(math.sqrt(float(p['max_blocks'])) + 1)/2)
        else:
            cap = float(p['max_window_m'])
        return max(cap, float(p['min_window_m']))

    def predict(self, number, x, y, altitude, block_m=None):
        """Register this frame's prior; return the search centre, altitude, radius and window.

        block_m: SCC block side of this frame (blocks mode); None when unknown.
        """
        p = self.p; cap = self.window_cap(block_m)
        if self.priors and (number <= self.priors[-1][0] or number - self.priors[-1][0] > p['reset_gap']):
            self.reset()
        self.priors.append((float(number), float(x), float(y), float(altitude)))
        s = self._smooth(number)
        state = dict(number=number, prior_x=float(x), prior_y=float(y), altitude=s['altitude'], frames=s['frames'])
        if self.fix is not None:
            gap = number - self.fix[0]
            limit = p['max_track_gap'] if self.confirmed else p['max_tentative_gap']
            if gap > limit:
                self.fix = self.prev_fix = None; self.confirmed = False
        if self.fix is not None:
            gap = number - self.fix[0]
            vx, vy = s['x_rate'], s['y_rate']
            if self.prev_fix is not None and 0 < self.fix[0]-self.prev_fix[0] <= p['max_track_gap']:
                d = self.fix[0]-self.prev_fix[0]; vx, vy = (self.fix[1]-self.prev_fix[1])/d, (self.fix[2]-self.prev_fix[2])/d
            cx, cy = (x + self.fix[3], y + self.fix[4]) if p['offset_model'] else (self.fix[1]+vx*gap, self.fix[2]+vy*gap)
            state.update(mode='track' if self.confirmed else 'tentative', x=cx, y=cy,
                         radius_m=p['track_radius_m'] + p['track_growth_m']*gap)
        else:
            mode = p['acquisition']
            if mode == 'hybrid': mode = 'smooth' if self.acquire_count % 2 else 'scan'
            if mode == 'smooth':
                state.update(mode='smooth', x=s['x'], y=s['y'], radius_m=s['radius_m'])
            else:
                tiles = scan_offsets(p['prior_error_m'], cap, p['scan_overlap'])
                dx, dy = tiles[(self.acquire_count//(2 if p['acquisition'] == 'hybrid' else 1)) % len(tiles)]
                bx, by = (x, y) if p['offset_model'] else (s['x'], s['y'])
                state.update(mode='scan', x=bx+dx, y=by+dy, radius_m=p['prior_error_m'])
            self.acquire_count += 1
        side = 2*(state['radius_m'] + p['margin_m'])
        state['window_m'] = float(min(max(side, p['min_window_m']), cap))
        state['window_cap_m'] = cap
        self.last = state
        return state

    def update(self, number, x, y, inliers):
        """Offer a localization result; returns True if accepted (tentative or track)."""
        s = self.last; p = self.p
        if s is None or s['number'] != number or inliers < p['min_fix_inliers']: return False
        # The truth is within prior_error_m of the raw prior by definition of the prior.
        if math.hypot(x - s['prior_x'], y - s['prior_y']) > p['prior_error_m'] + p['margin_m']: return False
        if s['mode'] in ('track', 'tentative'):
            if math.hypot(x - s['x'], y - s['y']) > s['radius_m']: return False
            self.confirmed = True
        else:
            # Acquisition: the result must lie inside the searched window; it starts a tentative track.
            if max(abs(x - s['x']), abs(y - s['y'])) > s['window_m']/2: return False
            self.prev_fix = None; self.confirmed = False
        self.prev_fix, self.fix = self.fix, (float(number), float(x), float(y), float(x - s['prior_x']), float(y - s['prior_y']))
        return True

    def output(self, x=None, y=None, accepted=False):
        """Position to report for the frame of the last predict(): (x, y, source)."""
        s = self.last
        if accepted: return x, y, 'fix'
        if s['mode'] in ('track', 'tentative'): return s['x'], s['y'], 'predicted'
        if x is not None: return x, y, 'unconfirmed_fix'
        return s['x'], s['y'], 'search_centre'
