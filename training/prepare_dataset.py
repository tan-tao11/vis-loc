"""Candidate protocol, strict geography split, footprint labels, no GT candidate insertion."""
import argparse
import copy
import math
import shutil
from pathlib import Path
import numpy as np
import shapely
from rasterio import Affine
from rasterio.enums import MaskFlags
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
from common import config, path, digest, read_json, write_json, input_fingerprint, require_real_ready, raster_files, sha256, generator_identity
from records import scan
from geometry import GeoRaster, Ground, intrinsics, camera_rotation, project_footprint, reference_angle
from preprocess import query_image, rotation_affine, intensity_u8

def label(query, valid_candidate, cfg):
    if query.is_empty or query.area <= 0: raise ValueError('Empty query footprint')
    coverage = query.intersection(valid_candidate).area/query.area
    ratio = valid_candidate.area/query.area
    distance = query.centroid.distance(valid_candidate.centroid) if not valid_candidate.is_empty else math.inf
    positive = (coverage >= cfg['positive_coverage'] and ratio <= cfg['max_area_ratio'] and
                distance <= cfg['max_center_distance_ratio']*math.sqrt(query.area))
    # High overlap but excessive scale remains ignored, never a false negative.
    value = 1 if positive else 0 if coverage <= cfg['negative_coverage'] else -1
    return value, float(coverage), float(ratio)

def satellite_intensity_check(satellite, spec, samples=1024):
    """Decimated band-1 statistics; reject maps whose configured range turns tiles white/black."""
    ds = satellite.ds; scale = max(1, math.ceil(max(ds.width, ds.height)/samples))
    shape_ = (max(1, ds.height//scale), max(1, ds.width//scale))
    a = ds.read(1, out_shape=shape_, masked=True)
    raw = np.asarray(a); valid = ~np.ma.getmaskarray(a) & np.isfinite(raw)
    report = {'dtype': str(raw.dtype), 'band_count': ds.count, 'nodata': ds.nodata, 'used_band': 1,
              'configured_range': spec.get('uint8' if raw.dtype == np.uint8 else 'uint16' if raw.dtype == np.uint16 else 'float')}
    if not valid.any(): raise ValueError(f'Satellite sample has no valid pixels: {report}')
    values = raw[valid].astype(np.float64)
    report['raw_percentiles'] = {k: float(np.percentile(values, k)) for k in (0.5, 50, 99.5)}
    u, mask = intensity_u8(raw, spec, valid)
    mapped = u[mask]
    report['mapped_white_fraction'] = float((mapped == 255).mean()) if mapped.size else None
    report['mapped_black_fraction'] = float((mapped == 0).mean()) if mapped.size else None
    if mapped.size and max(report['mapped_white_fraction'], report['mapped_black_fraction']) > 0.9:
        lo, hi = report['raw_percentiles'][0.5], report['raw_percentiles'][99.5]
        raise ValueError('Satellite intensity range saturates >90% of pixels (tiles render pure white/black, DINO features are '
                         f'meaningless). Set preprocess.intensity.satellite.{report["dtype"] if raw.dtype != np.uint8 else "uint8"} '
                         f'to the real value range, e.g. [{lo:g}, {hi:g}], and use the same image_value_range in inference. Stats: {report}')
    return report

class ValidIndex:
    """Low-resolution validity grid of the satellite, built once per prepare.

    A cell is valid only if every native pixel in it is valid (min reduction), so the
    grid is conservative. Replaces one native-resolution window read per candidate.
    """
    def __init__(self, raster, spec, target=4096, strip_pixels=64000000):
        ds = raster.ds; self.W, self.H = raster.width, raster.height; self.transform = raster.transform
        f = self.f = max(1, math.ceil(max(self.W, self.H)/target))
        h, w = math.ceil(self.H/f), math.ceil(self.W/f)
        kind = 'uint8' if ds.dtypes[0] == 'uint8' else 'uint16' if ds.dtypes[0] == 'uint16' else 'float' if 'float' in ds.dtypes[0] else None
        if kind is None or spec.get(kind) is None: raise ValueError(f'Explicit {kind or ds.dtypes[0]} intensity range required')
        self.trivial = (kind == 'uint8' and ds.nodata is None and spec.get('nodata') is None
                        and list(ds.mask_flag_enums[0]) == [MaskFlags.all_valid])
        if self.trivial:
            self.grid = np.ones((h, w), bool)
        else:
            self.grid = np.zeros((h, w), bool); k = max(1, strip_pixels//max(1, self.W*f))
            for r in range(0, h, k):
                rows = min(k, h-r); r0 = r*f; n = min(rows*f, self.H-r0)
                a = ds.read(1, window=Window(0, r0, self.W, n), masked=True)
                _, valid = intensity_u8(np.asarray(a), spec, ~np.ma.getmaskarray(a), mask_only=True)
                padded = np.ones((rows*f, w*f), bool); padded[:n, :self.W] = valid
                self.grid[r:r+rows] = padded.reshape(rows, f, w, f).all(axis=(1, 3))
        self._region = None

    def fraction(self, corners, n=16):
        """corners: (N, 4, 2) source-pixel parallelogram corners; returns sampled valid fraction."""
        u = (np.arange(n)+.5)/n; uu, vv = np.meshgrid(u, u); uu, vv = uu.ravel(), vv.ravel()
        c0, c1, c3 = corners[:, 0, None], corners[:, 1, None], corners[:, 3, None]
        pts = c0 + uu[None, :, None]*(c1-c0) + vv[None, :, None]*(c3-c0)
        col, row = pts[..., 0], pts[..., 1]
        inside = (col >= 0) & (row >= 0) & (col < self.W) & (row < self.H)
        ok = np.zeros(col.shape, bool)
        ok[inside] = self.grid[(row[inside]//self.f).astype(int), (col[inside]//self.f).astype(int)]
        return ok.mean(axis=1)

    def region(self):
        if self._region is None:
            t = self.transform
            rect = Polygon([t*(0, 0), t*(self.W, 0), t*(self.W, self.H), t*(0, self.H)])
            if self.trivial or self.grid.all(): self._region = rect
            else:
                tf = t*Affine.scale(self.f)
                polys = [shape(g) for g, v in shapes(self.grid.astype('uint8'), mask=self.grid, transform=tf) if v == 1]
                self._region = unary_union(polys).intersection(rect) if polys else Polygon()
        return self._region

def label_candidates(query, cands, corners, fractions, index, transform, cfg):
    """Vectorised equivalent of label() for all candidates of one query."""
    t = transform
    xs = t.a*corners[..., 0] + t.b*corners[..., 1] + t.c
    ys = t.d*corners[..., 0] + t.e*corners[..., 1] + t.f
    full = shapely.polygons(np.stack([xs, ys], -1))
    valid = full.copy(); partial = fractions < 1
    if partial.any(): valid[partial] = shapely.intersection(full[partial], index.region())
    qa = query.area
    if query.is_empty or qa <= 0: raise ValueError('Empty query footprint')
    coverage = shapely.area(shapely.intersection(valid, query))/qa
    ratio = shapely.area(valid)/qa
    dist = shapely.distance(shapely.centroid(valid), query.centroid)
    dist = np.where(shapely.is_empty(valid), np.inf, dist)
    positive = (coverage >= cfg['positive_coverage']) & (ratio <= cfg['max_area_ratio']) & (dist <= cfg['max_center_distance_ratio']*math.sqrt(qa))
    values = np.where(positive, 1, np.where(coverage <= cfg['negative_coverage'], 0, -1))
    centers = shapely.get_coordinates(shapely.centroid(full)).reshape(-1, 2)
    for k, cand in enumerate(cands):
        cand.update({'label': int(values[k]), 'coverage': float(coverage[k]), 'area_ratio': float(ratio[k]),
                     'valid_fraction': float(fractions[k]), 'center_map': centers[k].tolist()})
    return full

def subsample_train(cands, center, count, seed, identity):
    """Train split only: keep all positives plus `count` negatives (half nearest, half random).

    Ignored (-1) candidates never enter the loss and are dropped. Indices are re-numbered
    by list position downstream; original grid ids stay in cand['id'].
    """
    positives = [c for c in cands if c['label'] == 1]
    negatives = [c for c in cands if c['label'] == 0]
    if len(negatives) > count:
        near = sorted(negatives, key=lambda c: math.dist(c['center_map'], center))[:count//2]
        chosen = {id(c) for c in near}
        rest = [c for c in negatives if id(c) not in chosen]
        rng = np.random.default_rng(int(digest([seed, identity, 'train_negatives'])[:16], 16))
        for j in rng.choice(len(rest), count-len(near), replace=False): chosen.add(id(rest[j]))
        negatives = [c for c in negatives if id(c) in chosen]
    keep = {id(c) for c in positives+negatives}
    return [c for c in cands if id(c) in keep]

def assign_split(footprint, regions, buffer_m):
    hits = [name for name, poly in regions if poly.buffer(-buffer_m).covers(footprint)]
    if len(hits) != 1: raise ValueError('Footprint crosses split/buffer or no geographic region')
    return hits[0]

def sequence_assignments(pairs, split_config):
    assignments = split_config.get('sequence_splits')
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError('Sequence mode requires split.sequence_splits: {sequence_name: train/val/test}')
    if any(not isinstance(k, str) or not k or v not in ('train', 'val', 'test') for k, v in assignments.items()):
        raise ValueError('Invalid sequence_splits: names must be strings and values train/val/test')
    missing = sorted({p['sequence'] for p in pairs} - assignments.keys())
    if missing: raise ValueError('Unassigned sequences: ' + ', '.join(missing))
    # A sequence name is shared across DS/DV/NS/NV, independent of frame order.
    return {name: assignments[name] for name in sorted({p['sequence'] for p in pairs})}

def split_audit(groups, buffer_m, sequence_exclusive=True):
    # All candidate footprints, not just positive regions; sequence key includes data group.
    errors = []; seq = {}
    for g in groups:
        key = (g['group'], g['sequence'])
        if sequence_exclusive and key in seq and seq[key] != g['split']:
            errors.append({'reason': 'sequence_cross_split', 'sequence': key})
        seq[key] = g['split']
    by_split = {s: unary_union([shape(g['search_footprint']) for g in groups if g['split'] == s]) for s in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if not by_split[a].is_empty and not by_split[b].is_empty and by_split[a].buffer(buffer_m).intersects(by_split[b]):
            errors.append({'reason': 'candidate_geographic_leakage', 'splits': [a, b]})
    return errors

def protocol_record(record, cfg, seed, identity):
    r = copy.deepcopy(record)
    if cfg['protocol'] == 'original_prior': return r
    if cfg['protocol'] != 'simulated_prior': raise ValueError('Unknown prior protocol')
    from pyproj import Geod
    rng = np.random.default_rng(int(digest([seed, identity])[:16], 16))
    dist = rng.uniform(*cfg['simulated_horizontal_m']); azi = rng.uniform(0, 360)
    # Perturb original prior, not GT; this protocol never consults GT for candidate placement.
    r['droneLongitude'], r['droneLatitude'], _ = Geod(ellps='WGS84').fwd(r['droneLongitude'], r['droneLatitude'], azi, dist)
    r['droneAltitude'] += rng.uniform(-cfg['simulated_height_m'], cfg['simulated_height_m'])
    return r

def candidates(record, angle, satellite, c, ground):
    cc = c['candidates']; radius = cc['search_radius_m']
    inverse, size = rotation_affine(satellite.width, satellite.height, angle)
    forward = np.linalg.inv(inverse)
    xy = satellite.from_wgs.transform(record['droneLongitude'], record['droneLatitude'])
    col, row = ~satellite.transform * xy
    center = (forward @ [col, row, 1])[:2]
    # Convert map metre displacement through full affine, preserving rotated grid scaling.
    A = np.array([[satellite.transform.a, satellite.transform.b], [satellite.transform.d, satellite.transform.e]])
    axes = A @ inverse[:2, :2]; metres = np.linalg.norm(axes, axis=0)
    base_sizes = cc['sizes_m']
    offsets = cc.get('height_offsets_m', [0])
    if any(offsets):
        agl = record['droneAltitude']-ground.height(record['droneLongitude'], record['droneLatitude'])
        if agl <= 0: raise ValueError('Nonpositive prior AGL')
        # Without a calibrated height-to-size relation, require explicit nominal AGL.
        reference_agl = cc.get('reference_agl_m')
        if reference_agl is None: raise ValueError('reference_agl_m required for height scale protocol')
        sizes = sorted(set(round(s*(agl+dh)/reference_agl, 3) for s in base_sizes for dh in offsets if agl+dh > 0))
    else: sizes = base_sizes
    generated = 0
    for metres_side in sizes:
        wh = np.maximum(10, np.ceil(metres_side/metres).astype(int)); wh += wh % 2
        steps = np.maximum(1, (wh*(1-cc['overlap'])).astype(int))
        lower = np.floor(center-radius/metres).astype(int); upper = np.ceil(center+radius/metres).astype(int)
        # Grid is anchored to prior search-window origin, independent of GT.
        def starts(lo, hi, step, width):
            end = hi-width
            if end < lo: return []
            return sorted(set(list(range(lo, end+1, int(step)))+[int(end)]))
        for top in starts(lower[1], upper[1], steps[1], wh[1]):
            for left in starts(lower[0], upper[0], steps[0], wh[0]):
                generated += 1
                if generated > cc['max_per_query']: raise ValueError('Candidate count exceeds cap; narrow explicit protocol')
                box = [left, top, left+int(wh[0]), top+int(wh[1])]
                if box[0] < 0 or box[1] < 0 or box[2] > size[0] or box[3] > size[1]: continue
                yield {'id': str(generated), 'inverse': inverse.tolist(), 'box': box, 'scale_m': float(metres_side),
                       'angle_deg': angle, 'training_gt_inserted': False}

def prepare(c, limit=None):
    require_real_ready(c)
    if c['preprocess'].get('distortion') is not None:
        raise ValueError('Distortion correction not enabled in deployment_compatible mode; supply calibrated rectified imagery')
    out = path(c['output']); out.mkdir(parents=True, exist_ok=True)
    inspection = scan(c); write_json(out/'mapping_report.json', inspection)
    if inspection['errors']: raise ValueError('Mapping errors; inspect mapping_report.json before labels')
    if not inspection['pairs']: raise ValueError('No images')
    split_mode = c['split'].get('mode', 'geographic')
    regions_file, regions, assignments = None, [], {}
    if split_mode == 'geographic':
        regions_file = path(c['split']['regions_geojson'])
        regions = [(f['properties']['split'], shape(f['geometry'])) for f in read_json(regions_file)['features']]
        if any(s not in ('train', 'val', 'test') for s, _ in regions): raise ValueError('Invalid split region name')
    elif split_mode == 'sequence':
        # Assign before --limit and projection failures so a small check cannot reshuffle the split.
        assignments = sequence_assignments(inspection['pairs'], c['split'])
    else:
        raise ValueError('split.mode must be geographic or sequence')
    geo = copy.deepcopy(c['geo']); geo['dsm'] = str(path(geo['dsm'])) if geo['dsm'] else None
    satellite = GeoRaster(path(geo['satellite']), geo['authority']); ground = Ground(geo)
    if not satellite.crs.is_projected or any(abs(a.unit_conversion_factor-1) > 1e-8 for a in satellite.crs.axis_info[:2]):
        raise ValueError('Candidate/split metric coordinates require a metre-based projected satellite CRS')
    groups, failures = [], []
    pairs = inspection['pairs'][:limit] if limit else inspection['pairs']
    cc = c['candidates']; train_negatives = cc.get('train_negatives'); stride = int(cc.get('eval_query_stride') or 1)
    per_query = (train_negatives+64) if train_negatives else cc['max_per_query']
    estimated = len(pairs)*(per_query+1)*1024*4
    if estimated > shutil.disk_usage(out).free*.8: raise ValueError(f'Feature disk preflight upper bound {estimated} bytes exceeds 80% free space')
    stride_position = {}
    try:
        intensity = satellite_intensity_check(satellite, c['preprocess']['intensity']['satellite'])
        index = ValidIndex(satellite, c['preprocess']['intensity']['satellite'])
        for item in pairs:
            known = assignments.get(item['sequence']) if split_mode == 'sequence' else None
            if known in ('val', 'test') and stride > 1:
                key = (item['group'], item['sequence']); k = stride_position.get(key, 0); stride_position[key] = k+1
                if k % stride: continue
            try:
                _, box = query_image(item['image'], c['preprocess'])
                record = item['record']; R = camera_rotation(record, c['pose'])
                K = intrinsics(item['width'], item['height'], record, c['preprocess']['intrinsics_override'])
                footprint, center = project_footprint(record, R, K, box, geo, ground, satellite)
                prior = protocol_record(record, cc, c['seed'], item['id'])
                angle = reference_angle(prior, R, satellite, geo['normal_to_ellipsoid_m'])
                cs = list(candidates(prior, angle, satellite, c, ground))
                if cs:
                    inverse = np.array(cs[0]['inverse']); b = np.array([x['box'] for x in cs], float)
                    px = np.stack([b[:, [0, 2, 2, 0]], b[:, [1, 1, 3, 3]], np.ones((len(cs), 4))], -1) @ inverse.T
                    corners = px[..., :2]
                    fractions = index.fraction(corners)
                    keep = fractions >= cc['min_valid_fraction']
                    cs = [x for x, k in zip(cs, keep) if k]; corners, fractions = corners[keep], fractions[keep]
                full = label_candidates(footprint, cs, corners, fractions, index, satellite.transform, c['labels']) if cs else []
                search = shapely.union_all(list(full)+[footprint])
                split = (assign_split(search, regions, c['split']['buffer_m']) if split_mode == 'geographic'
                         else assignments[item['sequence']])
                if split == 'train' and train_negatives:
                    cs = subsample_train(cs, list(center), int(train_negatives), c['seed'], item['id'])
                groups.append({**item, 'split': split, 'protocol': cc['protocol'], 'rotation': R.tolist(),
                               'optical_ned': R[:, 2].tolist(), 'angle_deg': angle, 'K': K.tolist(), 'crop_box': list(box),
                               'query_footprint': mapping(footprint), 'ground_center_map': list(center),
                               'search_footprint': mapping(search), 'candidates': cs})
            except Exception as e:
                failures.append({'id': item['id'], 'reason': str(e), 'image': item['image']})
    finally:
        satellite.close(); ground.close()
    if split_mode == 'geographic' and stride > 1:
        # Geographic split is only known after projection; thin val/test frames afterwards.
        thinned, position = [], {}
        for g in groups:
            if g['split'] in ('val', 'test'):
                key = (g['group'], g['sequence']); k = position.get(key, 0); position[key] = k+1
                if k % stride: continue
            thinned.append(g)
        groups = thinned
    leakage = split_audit(groups, c['split']['buffer_m'], c['split']['sequence_exclusive'])
    split_warnings = []
    if split_mode == 'sequence':
        split_warnings = [x for x in leakage if x['reason'] == 'candidate_geographic_leakage']
        leakage = [x for x in leakage if x['reason'] != 'candidate_geographic_leakage']
        split_warnings.insert(0, {'reason': 'sequence_split_not_geographic_validation'})
        print('Sequence split: geographic independence is NOT enforced. Review split_warnings in preparation_report.json.')
    content_splits = {}
    for group in groups:
        content_hash = sha256(group['image'])
        previous = content_splits.get(content_hash)
        if previous and previous['split'] != group['split']:
            leakage.append({'reason': 'duplicate_image_cross_split', 'ids': [previous['id'], group['id']]})
        content_splits[content_hash] = group
    files = inspection['files']+raster_files(geo['satellite'])
    if regions_file is not None: files += [regions_file]
    if geo['dsm']: files += raster_files(geo['dsm'])
    fp = input_fingerprint(c, files)
    manifest = {'schema_version': 1, 'synthetic': c['synthetic'], 'fingerprint': fp, 'config': c, 'generator_code': generator_identity(),
                'total_matched': len(inspection['pairs']), 'processed': len(pairs), 'limited': limit is not None,
                'groups': groups, 'failures': failures, 'leakage': leakage,
                'split_mode': split_mode, 'sequence_splits': assignments, 'split_warnings': split_warnings}
    manifest['manifest_id'] = digest(manifest)
    write_json(out/'manifest.json', manifest, indent=None)
    write_json(out/'preparation_report.json', {'legal_queries': len(groups), 'failures': failures, 'leakage': leakage,
        'split_mode': split_mode, 'sequence_splits': assignments, 'split_warnings': split_warnings,
        'split_counts': {s: sum(g['split'] == s for g in groups) for s in ('train', 'val', 'test')},
        'covered_queries': sum(any(x['label'] == 1 for x in g['candidates']) for g in groups), 'estimated_feature_bytes': estimated,
        'candidates_per_split': {s: sum(len(g['candidates']) for g in groups if g['split'] == s) for s in ('train', 'val', 'test')},
        'satellite_intensity': intensity})
    if leakage: raise ValueError('Geographic split leakage; see preparation_report.json')
    if not groups: raise ValueError('All frames rejected; see preparation_report.json')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--limit', type=int)
    a = p.parse_args(); m = prepare(config(a.config), a.limit)
    print(f"Prepared {len(m['groups'])} legal queries; {len(m['failures'])} failures")
