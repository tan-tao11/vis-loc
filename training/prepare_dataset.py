"""Training manifest: deployed SCC search map/blocks (scc_inference) + geometric footprint labels.

Candidates are exactly the blocks the deployed infrared backend scores for this frame
(same map grid, window, rotation, block size, overlap and black-border filter). Labels
use the calibrated ground-truth footprint. No GT is ever inserted into the candidate set.
"""
import argparse
import copy
import math
import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as transform_geometry, unary_union
from common import config, path, digest, read_json, write_json, input_fingerprint, require_real_ready, raster_files, sha256, generator_identity
from records import scan
from geometry import GeoRaster, Ground, intrinsics, camera_rotation, project_footprint
from preprocess import query_image
from scc_inference import telemetry, reference_config, open_reference, camera_for, frame_retrieval, block_bounds, block_map_corners

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

def satellite_intensity_check(ref, samples=1024):
    """Band statistics after the deployed uint8 conversion; rejects ranges that make tiles white/black."""
    with rasterio.open(ref['image']) as ds:
        bands = sorted(set(int(b) for b in ref.get('image_bands', [1, 2, 3])))
        scale = max(1, math.ceil(max(ds.width, ds.height)/samples))
        a = ds.read(bands, out_shape=(len(bands), max(1, ds.height//scale), max(1, ds.width//scale)), masked=True)
        dtypes = [ds.dtypes[b-1] for b in bands]; count = ds.count
    raw = np.asarray(a, dtype=np.float64); valid = ~np.ma.getmaskarray(a) & np.isfinite(raw)
    value_range = ref.get('image_value_range')
    report = {'dtypes': dtypes, 'band_count': count, 'image_bands': ref.get('image_bands', [1, 2, 3]),
              'image_value_range': value_range}
    if any(d != 'uint8' for d in dtypes) and value_range is None:
        raise ValueError(f'Non-uint8 satellite needs inference.reference.image_value_range (same as naviParams): {report}')
    if not valid.any(): raise ValueError(f'Satellite sample has no valid pixels: {report}')
    values = raw[valid]
    report['raw_percentiles'] = {k: float(np.percentile(values, k)) for k in (0.5, 50, 99.5)}
    if value_range is not None:
        lo, hi = map(float, value_range); values = (values-lo)*(255/(hi-lo))
    mapped = np.clip(values, 0, 255).astype(np.uint8)
    report['mapped_white_fraction'] = float((mapped == 255).mean()); report['mapped_black_fraction'] = float((mapped == 0).mean())
    if max(report['mapped_white_fraction'], report['mapped_black_fraction']) > 0.9:
        lo, hi = report['raw_percentiles'][0.5], report['raw_percentiles'][99.5]
        raise ValueError('Satellite value range saturates >90% of pixels (tiles render pure white/black). Set '
                         f'inference.reference.image_value_range to the real range, e.g. [{lo:g}, {hi:g}], and use the same '
                         f'image_value_range in naviParams. Stats: {report}')
    return report

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

def search_prior(item, split, c, ref):
    """Search-centre telemetry and window for one frame (all splits).

    The deployed search centre comes from prior tracking/scanning, whose behaviour depends on the
    unknown prior error model. The retrieval model only matters once the truth is inside the window
    (tracking window or the acquisition tile that contains it), so every split uses: centre = GT +
    uniform offset within center_offset_m, altitude = GT +- altitude_error_m (block-size error),
    window = train_window_m (train) or eval_window_m (val/test, = acquisition tile, most negatives).
    GT only places the window; candidates are still the deployed grid and no GT block is inserted.
    """
    t = telemetry(item['record']); cc = c['candidates']; r = item['record']
    rng = np.random.default_rng(int(digest([c['seed'], item['id'], 'search_prior'])[:16], 16))
    gx, gy = ref.forward.transform(r['droneLongitude_gt'], r['droneLatitude_gt'], errcheck=True)
    d = float(cc.get('center_offset_m') or 0.)*math.sqrt(rng.random()); a = rng.uniform(0, 2*math.pi)
    t['lon'], t['lat'] = ref.inverse.transform(gx+d*math.cos(a), gy+d*math.sin(a), errcheck=True)
    e = float(cc.get('altitude_error_m') or 0.)
    t['altitude'] = float(r['droneAltitude_gt']) + rng.uniform(-e, e)
    prefix = 'train' if split == 'train' else 'eval'
    window = float(cc.get(prefix+'_window_m') or c['inference']['reference']['window_m'])
    mode = cc.get('window_mode', 'meters')
    if mode not in ('meters', 'blocks'): raise ValueError('candidates.window_mode must be meters or blocks')
    max_blocks = cc.get(prefix+'_max_blocks') if mode == 'blocks' else None
    return t, window, max_blocks

def frame_candidates(full_ref, item, t, c, footprint, window_m, max_blocks=None):
    """Deployed search blocks for one frame, labelled against the working-CRS footprint."""
    inf = c['inference']; spec = c['data']['groups'][item['group']]
    frame = frame_retrieval(full_ref, t, item['width'], item['height'], item['record']['isEO'], spec['isSlant'],
                            camera_for(c, item['group']), inf.get('pose_priori', 'yp'), inf.get('retrieval_overlap_percent', 50),
                            window_m=window_m, max_blocks=max_blocks, hard_cap_m=float(c['candidates'].get('hard_cap_m') or 8000.))
    polys, cands = [], []
    for k, mid in enumerate(frame['mids']):
        r0, r1, c0, c1 = block_bounds(mid, frame['side_px'])
        block = frame['ref_image'][r0:r1, c0:c1]
        fraction = float((block.reshape(block.shape[0], block.shape[1], -1) > 0).any(-1).mean()) if block.size else 0.
        poly = shapely.Polygon(block_map_corners(frame, mid)); polys.append(poly)
        value, coverage, ratio = label(footprint, poly, c['labels'])
        cands.append({'id': str(k), 'mid': [float(mid[0]), float(mid[1])], 'label': value, 'coverage': coverage,
                      'area_ratio': ratio, 'valid_fraction': fraction, 'scale_m': frame['side_m'],
                      'center_map': list(poly.centroid.coords[0]), 'training_gt_inserted': False})
    return frame, cands, polys

def prepare(c, limit=None):
    require_real_ready(c)
    if c['preprocess'].get('distortion') is not None:
        raise ValueError('Distortion correction not enabled; supply rectified imagery')
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
    cc = c['candidates']; eval_window = float(c['inference']['reference']['window_m'])
    train_negatives = cc.get('train_negatives'); stride = int(cc.get('eval_query_stride') or 1)
    intensity = satellite_intensity_check(reference_config(c, eval_window))
    geo = copy.deepcopy(c['geo']); geo['dsm'] = str(path(geo['dsm'])) if geo['dsm'] else None
    satellite = GeoRaster(path(geo['satellite']), geo['authority']); ground = Ground(geo)
    ref = open_reference(c, eval_window)
    working = CRS.from_user_input(ref.forward.target_crs)
    if working.equals(satellite.crs, ignore_axis_order=True): to_working = None
    else:
        tr = Transformer.from_crs(satellite.crs, working, always_xy=True)
        to_working = lambda geom: transform_geometry(tr.transform, geom)
    if to_working: regions = [(s, to_working(p)) for s, p in regions]
    groups, failures, stride_position = [], [], {}
    pairs = inspection['pairs'][:limit] if limit else inspection['pairs']
    try:
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
                if to_working:
                    footprint = to_working(footprint); center = tr.transform(*center)
                t, window, max_blocks = search_prior(item, known or 'val', c, ref)
                frame, cs, polys = frame_candidates(ref, item, t, c, footprint, window, max_blocks)
                search = shapely.union_all(polys+[footprint])
                split = (assign_split(search, regions, c['split']['buffer_m']) if split_mode == 'geographic'
                         else assignments[item['sequence']])
                if split == 'train' and known is None:   # geographic mode: split known only now
                    t, window, max_blocks = search_prior(item, 'train', c, ref)
                    frame, cs, polys = frame_candidates(ref, item, t, c, footprint, window, max_blocks)
                    search = shapely.union_all(polys+[footprint])
                grid_size = len(cs)
                if split == 'train' and train_negatives:
                    cs = subsample_train(cs, list(center), int(train_negatives), c['seed'], item['id'])
                groups.append({**item, 'split': split, 'protocol': 'scc_inference', 'is_slant': c['data']['groups'][item['group']]['isSlant'],
                               'telemetry': t, 'window_m': frame['window_m'], 'prior_source': 'gt_offset', 'side_px': frame['side_px'], 'side_m': frame['side_m'],
                               'grid_size': grid_size, 'yaw_deg': frame['yaw'], 'elevation_deg': frame['elevation'], 'agl_m': frame['agl'],
                               'rotation': R.tolist(), 'optical_ned': R[:, 2].tolist(), 'K': K.tolist(), 'crop_box': list(box),
                               'query_footprint': mapping(footprint), 'ground_center_map': list(center),
                               'search_footprint': mapping(search), 'candidates': cs})
            except Exception as e:
                failures.append({'id': item['id'], 'reason': str(e), 'image': item['image']})
    finally:
        satellite.close(); ground.close(); ref.close()
    if split_mode == 'geographic' and stride > 1:
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
    per_split = {s: [g for g in groups if g['split'] == s] for s in ('train', 'val', 'test')}
    write_json(out/'preparation_report.json', {'legal_queries': len(groups), 'failures': failures, 'leakage': leakage,
        'split_mode': split_mode, 'sequence_splits': assignments, 'split_warnings': split_warnings,
        'split_counts': {s: len(v) for s, v in per_split.items()},
        'covered_queries': {s: sum(any(x['label'] == 1 for x in g['candidates']) for g in v) for s, v in per_split.items()},
        'candidates_per_split': {s: sum(len(g['candidates']) for g in v) for s, v in per_split.items()},
        'window_m': {s: [round(min(g['window_m'] for g in v)), round(max(g['window_m'] for g in v))] if v else None for s, v in per_split.items()},
        'block_side_m': {s: [round(min(g['side_m'] for g in v), 1), round(max(g['side_m'] for g in v), 1)] if v else None for s, v in per_split.items()},
        'estimated_feature_bytes': sum(len(g['candidates'])+1 for g in groups)*1024*4, 'satellite_intensity': intensity})
    if leakage: raise ValueError('Geographic split leakage; see preparation_report.json')
    if not groups: raise ValueError('All frames rejected; see preparation_report.json')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--limit', type=int)
    a = p.parse_args(); m = prepare(config(a.config), a.limit)
    print(f"Prepared {len(m['groups'])} legal queries; {len(m['failures'])} failures")
