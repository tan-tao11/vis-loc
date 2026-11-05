"""Candidate protocol, strict geography split, footprint labels, no GT candidate insertion."""
import argparse
import copy
import math
import shutil
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
from common import config, path, digest, read_json, write_json, input_fingerprint, require_real_ready, raster_files, sha256, generator_identity
from records import scan
from geometry import GeoRaster, Ground, intrinsics, camera_rotation, project_footprint, reference_angle
from preprocess import query_image, rotation_affine, local_rotated_crop, valid_polygon

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
    estimated = len(pairs)*(c['candidates']['max_per_query']+1)*1024*4
    if estimated > shutil.disk_usage(out).free*.8: raise ValueError(f'Feature disk preflight upper bound {estimated} bytes exceeds 80% free space')
    try:
        for item in pairs:
            try:
                _, box = query_image(item['image'], c['preprocess'])
                record = item['record']; R = camera_rotation(record, c['pose'])
                K = intrinsics(item['width'], item['height'], record, c['preprocess']['intrinsics_override'])
                footprint, center = project_footprint(record, R, K, box, geo, ground, satellite)
                prior = protocol_record(record, c['candidates'], c['seed'], item['id'])
                angle = reference_angle(prior, R, satellite, geo['normal_to_ellipsoid_m'])
                cs = []
                for cand in candidates(prior, angle, satellite, c, ground):
                    _, mask, corners = local_rotated_crop(satellite, np.array(cand['inverse']), cand['box'], c['preprocess']['intensity']['satellite'], c['candidates']['max_window_pixels'], mask_only=True)
                    fraction = float(mask.mean())
                    if fraction < c['candidates']['min_valid_fraction']: continue
                    valid = valid_polygon(mask, cand['inverse'], cand['box'], satellite.transform)
                    value, coverage, ratio = label(footprint, valid, c['labels'])
                    full = Polygon([satellite.transform * tuple(p) for p in corners])
                    cand.update({'label': value, 'coverage': coverage, 'area_ratio': ratio, 'valid_fraction': fraction,
                                 'footprint': mapping(full), 'valid_footprint': mapping(valid), 'center_map': list(full.centroid.coords[0])})
                    cs.append(cand)
                search = unary_union([shape(x['footprint']) for x in cs]+[footprint])
                split = (assign_split(search, regions, c['split']['buffer_m']) if split_mode == 'geographic'
                         else assignments[item['sequence']])
                groups.append({**item, 'split': split, 'protocol': c['candidates']['protocol'], 'rotation': R.tolist(),
                               'optical_ned': R[:, 2].tolist(), 'angle_deg': angle, 'K': K.tolist(), 'crop_box': list(box),
                               'query_footprint': mapping(footprint), 'ground_center_map': list(center),
                               'search_footprint': mapping(search), 'candidates': cs})
            except Exception as e:
                failures.append({'id': item['id'], 'reason': str(e), 'image': item['image']})
    finally:
        satellite.close(); ground.close()
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
    write_json(out/'manifest.json', manifest)
    write_json(out/'preparation_report.json', {'legal_queries': len(groups), 'failures': failures, 'leakage': leakage,
        'split_mode': split_mode, 'sequence_splits': assignments, 'split_warnings': split_warnings,
        'split_counts': {s: sum(g['split'] == s for g in groups) for s in ('train', 'val', 'test')},
        'covered_queries': sum(any(x['label'] == 1 for x in g['candidates']) for g in groups), 'estimated_feature_bytes': estimated})
    if leakage: raise ValueError('Geographic split leakage; see preparation_report.json')
    if not groups: raise ValueError('All frames rejected; see preparation_report.json')
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--limit', type=int)
    a = p.parse_args(); m = prepare(config(a.config), a.limit)
    print(f"Prepared {len(m['groups'])} legal queries; {len(m['failures'])} failures")
