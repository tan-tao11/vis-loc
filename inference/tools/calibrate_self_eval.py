#!/usr/bin/env python3
"""Calibrate infrared.self_eval thresholds from inference logs and ground truth (stdlib only).

Inputs, one pair per sequence run (repeat --log / --data in the same order):
  --log   console log or timing.jsonl with the [NAV_TIMING] JSON of every frame
          (written when tuning.profile_timing is true, the default)
  --data  folder with the per-image TXT files of that run (truth = *_gt fields; the noisy copies made by
          add_prior_noise.py keep them)

Prints, for frames with ground truth:
  1. accuracy (error <= --tol m) of every output source and of the current selfEval / reported flags;
  2. for PnP fixes, the threshold sets that keep selfEval=1 precise (>= --precision) while covering the
     most correct frames, as a ready-to-paste self_eval line.

Example:
  python3 calibrate_self_eval.py --log logs/seq1.log --data data_noisy/day/IR/slant/sequence1 \
                                 --log logs/seq2.log --data data_noisy/day/IR/slant/sequence2
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_prior_noise import read_values, frame_number  # same TXT parser as the noise tool

M_PER_DEG = 111320.0


def truth(seq_dir):
    out = {}
    for txt in Path(seq_dir).glob('*.txt'):
        n = frame_number(txt)
        if n is None:
            continue
        v, _ = read_values(txt.read_text(encoding='utf-8-sig'))
        lon = v.get('droneLongitude_gt', v.get('droneLongtitude_gt')); lat = v.get('droneLatitude_gt'); alt = v.get('droneAltitude_gt')
        num = int(float(v.get('number', n)))
        if lon is not None and lat is not None:
            out[num] = (float(lon), float(lat), float(alt) if alt is not None else None)
    return out


def traces(log):
    for line in Path(log).read_text(encoding='utf-8', errors='replace').splitlines():
        i = line.find('{')
        if i < 0 or ('[NAV_TIMING]' not in line and not line.lstrip().startswith('{')):
            continue
        try:
            t = json.loads(line[i:])
        except ValueError:
            continue
        if isinstance(t, dict) and 'number' in t:
            yield t


def frames(logs, datas, use_3d):
    if len(logs) != len(datas):
        raise SystemExit('give one --data per --log, in the same order')
    rows = []
    for log, seq in zip(logs, datas):
        gt = truth(seq)
        for t in traces(log):
            out = t.get('output'); g = gt.get(int(t['number']))
            if not out or g is None:
                continue
            e = math.hypot((out[0]-g[0])*M_PER_DEG*math.cos(math.radians(g[1])), (out[1]-g[1])*M_PER_DEG)
            if use_3d and g[2] is not None:
                e = math.hypot(e, out[2]-g[2])
            rows.append(dict(err=e, source=t.get('output_source', 'fix'), reported=t.get('reported', True),
                             self_eval=int(out[3]), f=t.get('self_eval_features'), residual=t.get('track_residual_m')))
    return rows


def rule_ok(r, c):
    f = r['f']
    if r['source'] != 'fix' or not f:
        return False
    rep = f.get('reprojection')
    return (f['inliers'] >= c['min_inliers'] and (rep is not None and rep <= c['max_reprojection_px'])
            and f['block_dist'] <= c['max_block_ratio']*f['side_m'] and abs(f['dz']) <= c['max_dz_m']
            and (r['residual'] is None or r['residual'] <= c['max_track_residual_m'])
            and f['score_margin'] >= c['min_score_margin'] and f.get('agreeing', 1) >= c['min_agreeing'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log', action='append', required=True, help='inference log of one run, repeatable')
    p.add_argument('--data', action='append', required=True, help='sequence folder with the TXT truth of that run')
    p.add_argument('--tol', type=float, default=150., help='a frame is correct if its error <= tol (m)')
    p.add_argument('--precision', type=float, default=0.95, help='required share of correct frames among selfEval=1')
    p.add_argument('--use-3d', action='store_true', help='include the altitude error')
    a = p.parse_args()
    rows = frames(a.log, a.data, a.use_3d)
    if not rows:
        raise SystemExit('no frames with both an output and ground truth; check the log has [NAV_TIMING] lines')
    good = lambda rs: sum(r['err'] <= a.tol for r in rs)
    n = len(rows)
    print(f'{n} frames, {good(rows)} within {a.tol:.0f} m\n')
    print('1) by output source            frames   share   correct')
    for s in sorted({r['source'] for r in rows}):
        rs = [r for r in rows if r['source'] == s]
        print(f'   {s:28s} {len(rs):6d}   {len(rs)/n:5.2f}   {good(rs)/len(rs):7.2f}')
    for key in ('reported', 'self_eval'):
        rs = [r for r in rows if r[key]]
        if rs:
            print(f'   current {key}=1{"":15s} {len(rs):6d}   {len(rs)/n:5.2f}   {good(rs)/len(rs):7.2f}')
    fixes = [r for r in rows if r['source'] == 'fix' and r['f']]
    if not fixes:
        print('\nno PnP fixes with features in the log (self_eval_features missing)'); return
    total_good = good(fixes)
    grid = dict(min_inliers=[10, 20, 30, 50, 80, 120], max_reprojection_px=[2., 4., 8., 1e9], max_block_ratio=[0.25, 0.5, 0.75],
                max_dz_m=[50., 100., 150., 300.], max_track_residual_m=[50., 100., 200., 1e9], min_score_margin=[-1., 0., 0.01, 0.03],
                min_agreeing=[1, 2])
    best = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()):
        c = dict(zip(keys, combo))
        sel = [r for r in fixes if rule_ok(r, c)]
        if not sel:
            continue
        prec, cov = good(sel)/len(sel), good(sel)/max(1, total_good)
        best.append((prec >= a.precision, cov, prec, len(sel), c))
    best.sort(key=lambda b: (b[0], b[1], b[2]), reverse=True)
    print(f'\n2) PnP fixes: {len(fixes)}, correct {total_good}. Threshold sets (precision >= {a.precision} first, then coverage):')
    print('   precision coverage selected  thresholds')
    for ok, cov, prec, k, c in best[:8]:
        print(f'   {prec:9.2f} {cov:8.2f} {k:8d}  {c}')
    ok, cov, prec, k, c = best[0]
    if not ok:
        print(f'\n   no set reaches precision {a.precision}; the best one is shown first — consider selfEval=0 always')
    line = ', '.join(f'{k}: {v:g}' if v < 1e8 else f'{k}: 1000000' for k, v in c.items())
    print(f'\nsuggested:\n  self_eval: {{enabled: true, sources: [fix], {line}}}')


if __name__ == '__main__':
    main()
