"""selfEval of the official vNaviResult: 1.0 only when every reliability check of the frame passes.

Features of the chosen PnP fix (SCCBackend.localize) plus the output source and the tracking
residual (BaseBackend.localize_tracked). All thresholds come from infrared.self_eval; calibrate them
with tools/calibrate_self_eval.py on frames with ground truth.
"""
import math

DEFAULTS = dict(
    enabled=False,
    sources=['fix'],            # output sources that may get 1 (fix = PnP fix accepted by the tracker / consistency)
    min_inliers=30,             # PnP inliers of the chosen candidate
    max_reprojection_px=4.0,    # PnP reprojection error
    max_block_ratio=0.5,        # ground point of the fix within this fraction of the block side from the block centre
    max_dz_m=250.,              # |PnP altitude - prior altitude| (prior altitude error is up to 200 m)
    max_track_residual_m=100.,  # while tracking: |fix - predicted position|
    min_score_margin=0.0,       # retrieval score of the matched block minus the best score of any non-overlapping block
    min_agreeing=1)             # consistent candidates (incl. the chosen one) whose fixes lie within 50 m of it


def settings(cfg):
    c = dict(DEFAULTS, **(cfg or {}))
    if int(c['min_inliers']) < 5:
        raise ValueError('self_eval.min_inliers must be >=5')
    c['sources'] = list(c['sources'])
    return c


def decide(features, source, residual, cfg):
    """(value 0/1, failed checks). features is None when the frame has no PnP fix."""
    if not cfg['enabled']:
        return 0, ['disabled']
    failed = []
    if source not in cfg['sources']:
        failed.append('source')
    if not features:
        return 0, failed + ['no_fix']
    def bad(v): return v is None or not math.isfinite(v)
    if bad(features.get('inliers')) or features['inliers'] < cfg['min_inliers']: failed.append('inliers')
    if cfg.get('max_reprojection_px') is not None and (bad(features.get('reprojection')) or features['reprojection'] > cfg['max_reprojection_px']):
        failed.append('reprojection')
    if bad(features.get('block_dist')) or features['block_dist'] > cfg['max_block_ratio'] * features['side_m']: failed.append('block')
    if bad(features.get('dz')) or abs(features['dz']) > cfg['max_dz_m']: failed.append('altitude')
    if residual is not None and residual > cfg['max_track_residual_m']: failed.append('track')
    if bad(features.get('score_margin')) or features['score_margin'] < cfg['min_score_margin']: failed.append('retrieval')
    if features.get('agreeing', 0) < cfg['min_agreeing']: failed.append('agreement')
    return int(not failed), failed
