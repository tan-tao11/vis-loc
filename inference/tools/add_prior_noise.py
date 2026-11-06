#!/usr/bin/env python3
"""Add artificial prior errors to per-image TXT records for inference testing (stdlib only).

Layout:  <src>/day/IR/slant/<sequence>/xxxx1000.jpg + xxxx1000.txt, xxxx1001.jpg + xxxx1001.txt, ...
Only the PRIOR fields droneLongitude / droneLatitude / droneAltitude are changed; *_gt fields are kept
(and added from the original prior if a file has none, so the truth is never lost). Originals are never
modified: the whole tree is copied to <dst> (images copied, or hard-linked with --link).

Error models (one per sequence, reproducible with --seed):
  const   one fixed horizontal offset (random direction) and one altitude offset for the whole sequence
  drift   offset magnitude changes linearly from a start to an end value and its direction turns slowly
          (e.g. 3000 m -> 4000 m over the sequence, no jumps); altitude offset drifts the same way
  random  independent offset for every frame
Optional --jitter-m adds small independent per-frame noise on top of const/drift.

Example:
  python3 add_prior_noise.py --src data --dst data_noisy --mode drift --min-m 3000 --max-m 4000 --alt-m 200
A log <dst>/prior_noise_log.csv lists every frame's offset (east, north, up) and horizontal error.
"""
import argparse
import csv
import io
import math
import random
import re
import shutil
from pathlib import Path

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
PRIOR = {'lon': ('droneLongitude', 'droneLongtitude'), 'lat': ('droneLatitude',), 'alt': ('droneAltitude',)}
GT = {'lon': 'droneLongitude_gt', 'lat': 'droneLatitude_gt', 'alt': 'droneAltitude_gt'}
M_PER_DEG_LAT = 111320.0


def frame_number(path):
    m = re.search(r'(\d+)$', path.stem)
    return int(m.group(1)) if m else None


def is_csv(text):
    first = next((s for s in text.splitlines() if s.strip() and not s.lstrip().startswith(('#', '//'))), '')
    return (',' in first or '\t' in first) and 'number' in first and '=' not in first and ':' not in first


def read_values(text):
    """Prior and GT values of one record, plus the file style needed to rewrite it."""
    if is_csv(text):
        sep = '\t' if '\t' in text.splitlines()[0] else ','
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if r]
        if len(rows) != 2:
            raise ValueError('CSV TXT must have one header row and one record')
        return dict(zip([k.strip() for k in rows[0]], [v.strip() for v in rows[1]])), ('csv', sep)
    values = {}
    for k, v in re.findall(r'([A-Za-z_]\w*)\s*[:=]\s*([^,;\s{}]+)', text):
        values.setdefault(k, v.strip('"\''))
    sep = ':' if re.search(r'droneLatitude\s*:', text) else '='
    return values, ('kv', sep)


def get(values, names):
    for n in names:
        if n in values:
            return float(values[n]), n
    raise ValueError('missing ' + '/'.join(names))


def rewrite(text, style, changes, additions):
    """Replace prior values in place (keeping the file's format); append missing *_gt fields."""
    kind, sep = style
    if kind == 'csv':
        rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if r]
        keys = [k.strip() for k in rows[0]]
        for k, v in changes.items():
            rows[1][keys.index(k)] = v
        for k, v in additions.items():
            rows[0].append(k); rows[1].append(v)
        out = io.StringIO(); csv.writer(out, delimiter=sep, lineterminator='\n').writerows(rows)
        return out.getvalue()
    for k, v in changes.items():
        text, n = re.subn(r'(?<![\w])(' + re.escape(k) + r')(\s*[:=]\s*)([^,;\s{}]+)', lambda m: m.group(1) + m.group(2) + v, text, count=1)
        if n != 1:
            raise ValueError('cannot rewrite ' + k)
    if additions:
        text = text.rstrip('\n') + '\n' + ''.join(f'{k}{sep}{v}\n' for k, v in additions.items())
    return text


def sequence_offsets(n, args, rng):
    """List of (east, north, up) offsets in metres for n frames of one sequence."""
    lo, hi, alt = args.min_m, args.max_m, args.alt_m
    out = []
    if args.mode == 'random':
        for _ in range(n):
            r, a = rng.uniform(lo, hi), rng.uniform(0, 2 * math.pi)
            out.append((r * math.sin(a), r * math.cos(a), rng.uniform(-alt, alt)))
        return out
    if args.mode == 'const':
        r0 = r1 = rng.uniform(lo, hi); a0 = a1 = rng.uniform(0, 2 * math.pi); z0 = z1 = rng.uniform(-alt, alt)
    else:  # drift
        r0, r1 = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if args.drift_increasing:
            r0, r1 = sorted((r0, r1))
        a0 = rng.uniform(0, 2 * math.pi); a1 = a0 + math.radians(rng.uniform(-args.turn_deg, args.turn_deg))
        z0, z1 = rng.uniform(-alt, alt), rng.uniform(-alt, alt)
    for i in range(n):
        f = i / max(1, n - 1)
        r, a, z = r0 + (r1 - r0) * f, a0 + (a1 - a0) * f, z0 + (z1 - z0) * f
        jx, jy = (rng.gauss(0, args.jitter_m), rng.gauss(0, args.jitter_m)) if args.jitter_m > 0 else (0., 0.)
        out.append((r * math.sin(a) + jx, r * math.cos(a) + jy, z))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True, help='data root, e.g. data')
    p.add_argument('--dst', required=True, help='new data root (must not exist)')
    p.add_argument('--mode', choices=('const', 'drift', 'random'), default='drift')
    p.add_argument('--min-m', type=float, default=3704., help='smallest horizontal error, m (default 2 nmi)')
    p.add_argument('--max-m', type=float, default=5556., help='largest horizontal error, m (default 3 nmi)')
    p.add_argument('--alt-m', type=float, default=200., help='largest altitude error, m (+/-)')
    p.add_argument('--turn-deg', type=float, default=30., help='drift: max change of error direction over a sequence')
    p.add_argument('--drift-increasing', action='store_true', help='drift: error grows from start to end (e.g. 3000 -> 4000 m)')
    p.add_argument('--jitter-m', type=float, default=0., help='extra independent per-frame noise (std, m)')
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--link', action='store_true', help='hard-link images instead of copying (same drive only)')
    args = p.parse_args()
    if args.min_m < 0 or args.max_m < args.min_m:
        raise SystemExit('need 0 <= min-m <= max-m')
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if dst.exists():
        raise SystemExit(f'{dst} exists; choose a new --dst (originals are never overwritten)')
    sequences = {}
    for txt in sorted(src.rglob('*.txt')):
        if frame_number(txt) is None or not any(txt.with_suffix(s).is_file() or txt.with_suffix(s.upper()).is_file() for s in IMAGE_SUFFIXES):
            continue
        sequences.setdefault(txt.parent, []).append(txt)
    if not sequences:
        raise SystemExit('no image/TXT pairs found under ' + str(src))
    shutil.copytree(src, dst, copy_function=(lambda a, b: __import__('os').link(a, b)) if args.link else shutil.copy2)
    log = [('sequence_dir', 'frame', 'east_m', 'north_m', 'up_m', 'horizontal_error_m')]
    for seq_dir, txts in sorted(sequences.items()):
        txts.sort(key=frame_number)
        rel = seq_dir.relative_to(src).as_posix()
        rng = random.Random(f'{args.seed}:{rel}')
        offsets = sequence_offsets(len(txts), args, rng)
        for txt, (de, dn, du) in zip(txts, offsets):
            text = txt.read_text(encoding='utf-8-sig')
            values, style = read_values(text)
            lon, klon = get(values, PRIOR['lon']); lat, klat = get(values, PRIOR['lat']); alt, kalt = get(values, PRIOR['alt'])
            # truth: existing *_gt, else the original prior (the competition data has prior == truth)
            additions = {GT[k]: f'{v:.10f}' if k != 'alt' else f'{v:.7f}' for k, v in (('lon', lon), ('lat', lat), ('alt', alt))
                         if GT[k] not in values and not (k == 'lon' and 'droneLongtitude_gt' in values)}
            glon = float(values.get(GT['lon'], values.get('droneLongtitude_gt', lon)))
            glat = float(values.get(GT['lat'], lat)); galt = float(values.get(GT['alt'], alt))
            new_lat = glat + dn / M_PER_DEG_LAT
            new_lon = glon + de / (M_PER_DEG_LAT * math.cos(math.radians(glat)))
            changes = {klon: f'{new_lon:.10f}', klat: f'{new_lat:.10f}', kalt: f'{galt + du:.7f}'}
            out = dst / txt.relative_to(src)
            if args.link:
                out.unlink()   # never write through a hard link into the original TXT
            out.write_text(rewrite(text, style, changes, additions), encoding='utf-8')
            log.append((rel, frame_number(txt), f'{de:.1f}', f'{dn:.1f}', f'{du:.1f}', f'{math.hypot(de, dn):.1f}'))
    with open(dst / 'prior_noise_log.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(log)
    errs = [float(r[5]) for r in log[1:]]
    print(f'{len(sequences)} sequences, {len(errs)} frames written to {dst}; mode={args.mode}, '
          f'horizontal error {min(errs):.0f}-{max(errs):.0f} m; log: prior_noise_log.csv')


if __name__ == '__main__':
    main()
