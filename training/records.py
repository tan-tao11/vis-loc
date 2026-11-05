"""Explicit (group, sequence, frame) identity. Never replicate a record."""
import csv
import io
import re
from pathlib import Path
from PIL import Image
import numpy as np
from common import path

ALIASES = {'droneLongtitude': 'droneLongitude', 'droneLongtitude_gt': 'droneLongitude_gt', 'ccdSize': 'ccdsize'}
REQUIRED = ['number', 'droneLongitude', 'droneLatitude', 'droneAltitude',
            'droneLongitude_gt', 'droneLatitude_gt', 'droneAltitude_gt',
            'innsYaw', 'innsPitch', 'innsRoll', 'sensorYaw', 'sensorPitch',
            'sensorRoll', 'hfov', 'vfov', 'isEO']

def put(record, key, value, source, line):
    key = ALIASES.get(key, key)
    value = value.strip().strip('"\'')
    try:
        v = float(value)
        if not np.isfinite(v):
            raise ValueError('Nonfinite field')
    except ValueError:
        if key != 'sequence':
            raise ValueError(f'{source}:{line}: invalid numeric {key}={value}')
        v = value
    if key in record and record[key] != v:
        raise ValueError(f'{source}:{line}: conflicting duplicate/alias {key}')
    record[key] = v
    record.setdefault('_lines', {})[key] = line

def parse_records(filename, fmt='auto'):
    p = Path(filename)
    lines = p.read_text(encoding='utf-8-sig').splitlines()
    useful = [(n, s.strip()) for n, s in enumerate(lines, 1) if s.strip() and not s.lstrip().startswith(('#', '//'))]
    if not useful:
        raise ValueError(f'{p}: empty telemetry')
    first = useful[0][1]
    if fmt == 'auto':
        fmt = 'csv' if (',' in first or '\t' in first) and 'number' in first and '=' not in first and ':' not in first else 'kv'
    records = []
    if fmt == 'csv':
        sep = '\t' if '\t' in first else ','
        rows = list(csv.reader(io.StringIO('\n'.join(s for _, s in useful)), delimiter=sep))
        keys = rows[0]
        for (line, _), vals in zip(useful[1:], rows[1:]):
            if len(vals) != len(keys):
                raise ValueError(f'{p}:{line}: column count mismatch')
            r = {'_source': str(p.resolve()), '_line': line}
            for k, v in zip(keys, vals):
                put(r, k.strip(), v, p, line)
            records.append(r)
    elif fmt == 'kv':
        r = {}
        for n, s in useful:
            if s in ('---', '{', '}', '[record]'):
                if s in ('---', '[record]') and r:
                    records.append(r); r = {}
                continue
            pairs = re.findall(r'([A-Za-z_]\w*)\s*[:=]\s*([^,;\s{}]+)', s)
            if not pairs:
                raise ValueError(f'{p}:{n}: unknown record syntax; configure parser')
            for k, v in pairs:
                if k == 'number' and 'number' in r:
                    records.append(r); r = {}
                r.setdefault('_source', str(p.resolve())); r.setdefault('_line', n)
                put(r, k, v, p, n)
        if r:
            records.append(r)
    else:
        raise ValueError(f'Unknown record_format {fmt}')
    return records, fmt

def validate_record(r, g):
    missing = [k for k in REQUIRED if k not in r]
    if missing:
        raise ValueError('Missing per-frame fields: ' + ', '.join(missing))
    if r['number'] < 0 or r['number'] != int(r['number']):
        raise ValueError('number must be nonnegative integer')
    for k in ('droneLongitude', 'droneLongitude_gt'):
        if not -180 <= r[k] <= 180: raise ValueError(k + ' out of range')
    for k in ('droneLatitude', 'droneLatitude_gt'):
        if not -90 <= r[k] <= 90: raise ValueError(k + ' out of range')
    for k in ('hfov', 'vfov'):
        if not 0 < r[k] < 179: raise ValueError(k + ' out of range')
    expected = {'imgType': 0, 'dayOrNight': g['dayOrNight'], 'isSlant': g['isSlant'], 'isEO': g['expected_isEO']}
    for k, val in expected.items():
        if k in r and r[k] != val:
            raise ValueError(f'{k} conflict: record={r[k]}, group={val}')
        if k != 'isEO': r[k] = val

def scan(c):
    d = c['data']; root = path(d['root']); rx = re.compile(d['image_regex'], re.I)
    pairs, errors, formats, all_files = [], [], {}, []
    def err(kind, **kw): errors.append({'reason': kind, **kw})
    for group, g in d['groups'].items():
        directory = root / g['directory']
        if not directory.is_dir():
            err('missing_directory', path=str(directory), group=group); continue
        images = {}
        for p in sorted(directory.rglob('*')):
            if p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'): continue
            all_files.append(p)
            m = rx.search(p.relative_to(directory).as_posix())
            if not m:
                err('image_name_unmatched', path=str(p)); continue
            key = (group, m['sequence'], int(m['frame']))
            if key in images:
                err('duplicate_image', key=list(key), paths=[str(images[key]), str(p)]); continue
            images[key] = p
        recmap = {}
        sources = [p.with_suffix('.txt') for p in images.values()] if d['per_image_txt'] else [root / g['records']]
        for src in sources:
            if not src.is_file(): err('missing_txt', path=str(src)); continue
            all_files.append(src)
            try:
                records, fmt = parse_records(src, d['record_format']); formats[str(src)] = {'format': fmt, 'records': len(records)}
                if d['per_image_txt'] and len(records) != 1: raise ValueError('Per-image TXT must contain exactly one record')
                for r in records:
                    validate_record(r, g)
                    seq = r.get('sequence', d['sequence_by_file'].get(src.relative_to(root).as_posix()))
                    if d['per_image_txt']:
                        m = rx.search(src.with_suffix('.png').relative_to(directory).as_posix())
                        if m is None: raise ValueError('Per-image sequence cannot be resolved')
                        if seq is not None and seq != m['sequence']: raise ValueError('sequence conflict')
                        seq = m['sequence']
                        if int(r['number']) != int(m['frame']): raise ValueError('Per-image frame conflict')
                    if not seq: raise ValueError('sequence missing: explicit sequence_by_file mapping required')
                    key = (group, str(seq), int(r['number']))
                    if key in recmap: raise ValueError(f'Duplicate telemetry identity {key}')
                    recmap[key] = r
            except (ValueError, OSError) as e:
                err('record_error', path=str(src), detail=str(e))
        for key, p in images.items():
            if key not in recmap: err('missing_record', path=str(p), key=list(key)); continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    if w < 2 or h < 2: raise ValueError('Image too small')
                    im.verify()
                pairs.append({'id': '/'.join(map(str, key)), 'group': group, 'sequence': key[1], 'number': key[2],
                              'image': str(p.resolve()), 'width': w, 'height': h, 'record': recmap[key]})
            except Exception as e: err('image_read_failure', path=str(p), detail=str(e))
        for key, r in recmap.items():
            if key not in images: err('unmatched_record', key=list(key), source=r['_source'], line=r['_line'])
    return {'pairs': pairs, 'errors': errors, 'formats': formats, 'files': [str(p.resolve()) for p in all_files]}
