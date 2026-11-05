"""Read-only pre-calibration diagnostics. Never creates labels or confirms calibration."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import html
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry
from common import ROOT, config, digest, path, sha256, write_json
from records import scan
from geometry import (checked_rotation, euler_zyx, camera_rotation, intrinsics,
                      CameraFrame, Ground, GeoRaster, intersect, project_footprint, reference_angle)
from preprocess import center_crop_box, query_image, intensity_u8
from outputs import fresh_directory, write_csv

GEOD = Geod(ellps='WGS84')


def analytic_checks():
    """Independent limiting cases; passing these does NOT validate a real sensor."""
    results = {
        'positive_yaw_turns_north_to_east': bool(np.allclose(euler_zyx(90, 0, 0) @ [1, 0, 0], [0, 1, 0])),
        'negative_pitch_turns_forward_down': bool(np.allclose(euler_zyx(0, -90, 0) @ [1, 0, 0], [0, 0, 1]))}
    frame = CameraFrame(87, 42, 200, 0)
    north, east, down = [frame.position(v) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    results['ned_basis_directions'] = bool(north[1] > 42 and east[0] > 87 and down[2] < 200)
    ground = Ground({'dsm': None, 'flat_fallback': True, 'flat_ground_normal_m': 100})
    nadir = intersect(frame, np.array([0., 0., 1.]), ground, 5, 300)
    tilt = intersect(frame, np.array([.5, 0., math.sqrt(.75)]), ground, 5, 300)
    distance = GEOD.inv(87, 42, tilt[0], tilt[1])[2]
    results['nadir_hits_platform_horizontal_position'] = bool(np.allclose(nadir, [87, 42, 100], atol=1e-6))
    results['small_area_tilt_matches_height_tan_angle'] = abs(distance-100/math.sqrt(3)) < .02
    return results


def static_checks(c):
    issues = []
    def add(level, field, detail): issues.append({'level': level, 'field': field, 'detail': detail})
    def finite_number(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    p, g, pre = c['pose'], c['geo'], c['preprocess']
    if p['euler_order'] != 'ZYX': add('error', 'pose.euler_order', 'Runtime implements explicit ZYX only')
    for key in ('eo_inns_from_cam', 'inns_from_mount', 'sensor_from_cam'):
        try: checked_rotation(p[key])
        except (ValueError, TypeError) as e: add('error', 'pose.'+key, str(e))
    try:
        axis = np.asarray(p['eo_inns_from_cam'], dtype=float)[:, 2]
        if not np.allclose(axis, [1, 0, 0], atol=1e-7):
            add('error', 'pose.eo_inns_from_cam', 'EO optical axis must be parallel to INS +X per supplied interface')
    except (ValueError, IndexError, TypeError): pass
    for key in ('normal_to_ellipsoid_m', 'dsm_to_platform_offset_m'):
        value = g.get(key)
        if value is None: add('missing', 'geo.'+key, 'Vertical datum conversion has not been supplied; no implicit zero')
        elif not finite_number(value): add('error', 'geo.'+key, 'Must be a finite number')
    for key in ('dsm_unit_to_m', 'ray_step_m', 'max_ray_m'):
        if not finite_number(g[key]) or g[key] <= 0: add('error', 'geo.'+key, 'Must be finite and positive')
    if all(finite_number(g[k]) for k in ('max_ray_m', 'ray_step_m')) and g['max_ray_m'] <= g['ray_step_m']:
        add('error', 'geo.max_ray_m', 'Must exceed ray_step_m')
    if not isinstance(g['boundary_samples_per_edge'], int) or g['boundary_samples_per_edge'] < 2:
        add('error', 'geo.boundary_samples_per_edge', 'At least two samples per edge required')
    if pre.get('distortion') is not None: add('error', 'preprocess.distortion', 'Current pipeline requires rectified imagery; distortion is not applied')
    K = pre.get('intrinsics_override')
    if K is not None:
        try:
            K = intrinsics(640, 512, {'hfov': 60, 'vfov': 45}, K)
            if not np.allclose(K[2], [0, 0, 1]) or abs(np.linalg.det(K)) < 1e-12:
                raise ValueError('Expected nonsingular standard pinhole K with last row [0,0,1]')
        except (ValueError, TypeError) as e: add('error', 'preprocess.intrinsics_override', str(e))
    if c['split'].get('mode', 'geographic') == 'geographic' and not c['split'].get('regions_geojson'):
        add('warning', 'split.regions_geojson', 'Geography split is needed for training, but not for this diagnostic')
    return issues


def load_controls(filename):
    if filename is None: return []
    rows, seen = [], set()
    with open(filename, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = {'id', 'pixel_x', 'pixel_y', 'longitude', 'latitude', 'role'}
        if not fields.issubset(reader.fieldnames or []): raise ValueError('Control CSV requires '+', '.join(sorted(fields)))
        for line, raw in enumerate(reader, 2):
            if any(raw.get(k) is None for k in fields): raise ValueError(f'Missing control value at line {line}')
            row = {'id': raw['id'].strip(), 'role': raw['role'].strip(), 'line': line}
            if not row['id'] or row['role'] not in ('calibration', 'check'): raise ValueError(f'Invalid control id/role at line {line}')
            for key in ('pixel_x', 'pixel_y', 'longitude', 'latitude'):
                row[key] = float(raw[key])
                if not np.isfinite(row[key]): raise ValueError(f'Nonfinite control value at line {line}')
            if not -180 <= row['longitude'] <= 180 or not -90 <= row['latitude'] <= 90:
                raise ValueError(f'Control coordinates outside lon/lat ranges at line {line}')
            identity = (row['id'], row['pixel_x'], row['pixel_y'])
            if identity in seen: raise ValueError(f'Duplicate control pixel at line {line}')
            seen.add(identity); rows.append(row)
    return rows


def control_error(control, item, R, K, frame, ground, geo):
    """Pixels use original-image coordinates; observed landmarks are independent data."""
    result = dict(control)
    try:
        x, y = control['pixel_x'], control['pixel_y']
        if not 0 <= x < item['width'] or not 0 <= y < item['height']:
            raise ValueError('Control pixel outside original image; do not pass resized/cropped coordinates')
        ray = R @ np.linalg.solve(K, [x, y, 1.])
        hit = intersect(frame, ray, ground, geo['ray_step_m'], geo['max_ray_m'])
        bearing, _, distance = GEOD.inv(control['longitude'], control['latitude'], hit[0], hit[1])
        result.update(predicted_lon_lat_normal=hit.tolist(), error_m=float(distance),
            error_east_m=float(distance*math.sin(math.radians(bearing))),
            error_north_m=float(distance*math.cos(math.radians(bearing))))
    except (ValueError, np.linalg.LinAlgError) as e: result['error'] = str(e)
    return result


def summarize_controls(rows, tolerance=None):
    good = [r for r in rows if 'error_m' in r]
    out = {'points': len(rows), 'frames': len({r['id'] for r in rows}), 'evaluated': len(good),
           'failed_or_unmatched': len(rows)-len(good), 'tolerance_m': tolerance}
    if good:
        e = np.array([r['error_m'] for r in good])
        out.update(median_m=float(np.median(e)), p95_m=float(np.percentile(e, 95)), max_m=float(e.max()),
            rmse_m=float(np.sqrt(np.mean(e*e))), bias_east_m=float(np.mean([r['error_east_m'] for r in good])),
            bias_north_m=float(np.mean([r['error_north_m'] for r in good])))
    out['all_within_tolerance'] = (bool(rows) and len(rows) == len(good) and all(r['error_m'] <= tolerance for r in good)) if tolerance is not None else None
    return out


def select_frames(pairs, maximum, controls):
    requested = {r['id'] for r in controls}
    selected = [p for p in pairs if p['id'] in requested]
    buckets = defaultdict(list)
    for p in pairs:
        if p['id'] not in requested: buckets[p['group']].append(p)
    # Evenly spaced samples within each group, interleaved across groups.
    buckets = {k: [v[i] for i in np.linspace(0, len(v)-1, min(maximum, len(v)), dtype=int)] for k, v in buckets.items()}
    for i in range(max((len(v) for v in buckets.values()), default=0)):
        for k in sorted(buckets):
            if len(selected) >= max(maximum, len(requested)): return selected
            if i < len(buckets[k]): selected.append(buckets[k][i])
    return selected


def inspect_frame(item, c, satellite, ground, controls, position_source):
    r = dict(item['record']); suffix = '_gt' if position_source == 'gt' else ''
    for field in ('droneLongitude', 'droneLatitude', 'droneAltitude'): r[field+'_gt'] = item['record'][field+suffix]
    geo = c['geo']; row = {'id': item['id'], 'group': item['group'], 'image': item['image'],
        'position_source': position_source, 'isEO': r['isEO'], 'errors': [], 'warnings': [], 'controls': [], 'hits': {}}
    R = camera_rotation(r, c['pose']); K = intrinsics(item['width'], item['height'], r, c['preprocess']['intrinsics_override'])
    frame = CameraFrame(r['droneLongitude_gt'], r['droneLatitude_gt'], r['droneAltitude_gt'], geo['normal_to_ellipsoid_m'])
    box = center_crop_box(item['width'], item['height']); l, t, right, bottom = box
    row.update(rotation=R.tolist(), K=K.tolist(), crop_box=list(box), optical_ned=R[:, 2].tolist(),
        image_right_ned=R[:, 0].tolist(), image_down_ned=R[:, 1].tolist(),
        optical_depression_deg=float(np.degrees(np.arcsin(np.clip(R[2, 2], -1, 1)))),
        camera_lon_lat_normal=[r['droneLongitude_gt'], r['droneLatitude_gt'], r['droneAltitude_gt']])
    try:
        row['agl_at_camera_m'] = r['droneAltitude_gt']-ground.height(r['droneLongitude_gt'], r['droneLatitude_gt'])
        if row['agl_at_camera_m'] <= 0: row['errors'].append('Camera at/below ground; check heights, units and datum')
        row['reference_rotation_deg'] = reference_angle(r, R, satellite, geo['normal_to_ellipsoid_m'])
    except ValueError as e: row['errors'].append(str(e))
    points = {'center': ((l+right)/2, (t+bottom)/2), 'right': ((l+3*right)/4, (t+bottom)/2),
              'down': ((l+right)/2, (t+3*bottom)/4), 'top_left': (l, t), 'top_right': (right, t),
              'bottom_right': (right, bottom), 'bottom_left': (l, bottom)}
    for name, uv in points.items():
        try: row['hits'][name] = intersect(frame, R @ np.linalg.solve(K, [*uv, 1.]), ground, geo['ray_step_m'], geo['max_ray_m']).tolist()
        except ValueError as e: row['errors'].append(name+': '+str(e))
    try:
        footprint, _ = project_footprint(r, R, K, box, geo, ground, satellite)
        row['footprint_wgs84'] = mapping(transform_geometry(satellite.to_wgs.transform, footprint))
        row['footprint_area_m2'] = float(footprint.area)
    except ValueError as e: row['errors'].append('footprint: '+str(e))
    for point in controls: row['controls'].append(control_error(point, item, R, K, frame, ground, geo))
    if 'center' in row['hits']:
        llh = row['hits']['center']
        distance = GEOD.inv(r['droneLongitude_gt'], r['droneLatitude_gt'], llh[0], llh[1])[2]
        row['platform_to_image_center_horizontal_m'] = float(distance)
        if row['optical_depression_deg'] < 10: row['warnings'].append('Shallow view: height/attitude errors can cause large horizontal displacement')
        # Independent local-flat approximation, explicitly not a real accuracy residual.
        if R[2, 2] > 1e-5:
            height = r['droneAltitude_gt']-ground.height(llh[0], llh[1])
            row['flat_local_optical_horizontal_estimate_m'] = float(height*np.linalg.norm(R[:2, 2])/R[2, 2])
    return row


def draw_overlay(item, row, c, satellite, filename):
    locations = [v[:2] for v in row['hits'].values()]+[row['camera_lon_lat_normal'][:2]]
    if 'footprint_wgs84' in row: locations += row['footprint_wgs84']['coordinates'][0]
    for p in row['controls']:
        locations.append([p['longitude'], p['latitude']])
        if 'predicted_lon_lat_normal' in p: locations.append(p['predicted_lon_lat_normal'][:2])
    def pixel(ll): return np.array(~satellite.transform * satellite.from_wgs.transform(*ll[:2]))
    pixels = np.array([pixel(ll) for ll in locations])
    if not np.isfinite(pixels).all(): raise ValueError('Control/projection cannot be represented in satellite CRS')
    low, high = pixels.min(0), pixels.max(0); margin = np.maximum((high-low)*.12, 25)
    low = np.maximum(np.floor(low-margin), 0); high = np.minimum(np.ceil(high+margin), [satellite.width, satellite.height])
    width, height = (high-low).astype(int)
    if min(width, height) <= 0: raise ValueError('No satellite window overlaps projected geometry')
    scale = min(940/width, 580/height); ow, oh = max(1, round(width*scale)), max(1, round(height*scale))
    a = satellite.ds.read(1, window=Window(*low, width, height), out_shape=(oh, ow), masked=True, resampling=Resampling.nearest)
    gray, valid = intensity_u8(a.data, c['preprocess']['intensity']['satellite'], ~np.ma.getmaskarray(a))
    rgb = np.repeat(gray[..., None], 3, axis=-1); rgb[~valid] = [110, 35, 35]
    canvas = Image.new('RGB', (1360, 780), '#101924'); canvas.paste(Image.fromarray(rgb), (400, 65))
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    draw.text((20, 15), ('SYNTHETIC / ' if c.get('synthetic') else 'UNVERIFIED GEOMETRY / ')+row['id'], fill='white', font=font)
    query, _ = query_image(item['image'], c['preprocess'])
    canvas.paste(Image.fromarray(query).convert('RGB').resize((350, 350)), (20, 65))
    def point(ll):
        xy = (pixel(ll)-low)*np.array([ow/width, oh/height])+[400, 65]
        return tuple(map(float, xy))
    def dot(ll, color, label='', label_offset=(6, 2)):
        x, y = point(ll); draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
        if label: draw.text((x+label_offset[0], y+label_offset[1]), label, fill=color, font=font)
    if 'footprint_wgs84' in row:
        draw.line([point(ll) for ll in row['footprint_wgs84']['coordinates'][0]], fill='#27e4ef', width=3)
    dot(row['camera_lon_lat_normal'], '#ff9e42', 'platform', (6, -24))
    if 'center' in row['hits']:
        center = row['hits']['center']; dot(center, '#ffffff', 'center')
        for key, color in (('right', '#ffdb40'), ('down', '#eb7dff')):
            if key in row['hits']:
                draw.line([point(center), point(row['hits'][key])], fill=color, width=3); dot(row['hits'][key], color, key)
        north = GEOD.fwd(center[0], center[1], 0, 100)[:2]
        vector = np.array(point(north))-point(center)
        if np.linalg.norm(vector) > 0:
            vector *= 45/np.linalg.norm(vector); start = np.array([1295., 110.]); end = start+vector
            draw.line([tuple(start), tuple(end)], fill='#70ff87', width=3); draw.text(tuple(end), 'N', fill='#70ff87', font=font)
    for i, p in enumerate(row['controls'], 1):
        observed = [p['longitude'], p['latitude']]; dot(observed, '#57ff66', str(i))
        if 'predicted_lon_lat_normal' in p:
            predicted = p['predicted_lon_lat_normal']; dot(predicted, '#ff5050')
            draw.line([point(observed), point(predicted)], fill='#ff5050', width=2)
    facts = ['Query crop (same as training)', f"position source: {row['position_source']}; isEO={row['isEO']}",
        f"optical down angle: {row['optical_depression_deg']:.2f} deg",
        f"AGL: {row.get('agl_at_camera_m', float('nan')):.2f} m",
        f"area: {row.get('footprint_area_m2', float('nan')):.1f} m2",
        'Cyan: predicted footprint', 'Yellow / magenta: image right / down',
        'Green: observed control; red: projected', 'Orange: platform, NOT image center']
    for i, text in enumerate(facts): draw.text((20, 435+i*25), text, fill='white', font=font)
    for i, message in enumerate((row['errors']+row['warnings'])[:3]): draw.text((400, 665+i*27), message[:100], fill='#ffc06e', font=font)
    canvas.save(filename)


def run(c, output, max_frames=16, controls_file=None, tolerance=None, position_source='gt'):
    if max_frames < 1 or position_source not in ('gt', 'prior'): raise ValueError('Invalid frame limit or position source')
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0): raise ValueError('Tolerance must be finite and positive')
    out = fresh_directory(output)
    report = {'schema_version': 1, 'purpose': 'fixed_geometry_diagnostic_only', 'synthetic': c.get('synthetic', False),
        'calibration_confirmed_by_this_tool': False, 'config_modified': False, 'position_source': position_source,
        'config_sha256': digest(c), 'code_sha256': {name: sha256(ROOT/name) for name in ('validate_geometry.py', 'geometry.py', 'records.py', 'preprocess.py')},
        'analytic_checks': analytic_checks(), 'issues': static_checks(c), 'frames': [], 'controls': [],
        'scope_note': 'Platform GT is not an observed ground landmark. No automatic calibration, fitting, training labels or train/val/test selection.'}
    try:
        controls = load_controls(controls_file)
        if controls_file: report['control_csv_sha256'] = sha256(controls_file)
    except (ValueError, OSError, TypeError) as e:
        controls = []
        report['issues'].append({'level': 'error', 'field': 'controls_csv', 'detail': str(e)})
    try: inspected = scan(c)
    except (ValueError, OSError, KeyError, TypeError) as e:
        inspected = {'pairs': [], 'errors': [str(e)]}
        report['issues'].append({'level': 'error', 'field': 'image_txt_mapping', 'detail': str(e)})
    write_json(out/'mapping_report.json', inspected)
    report['mapping'] = {'matched': len(inspected['pairs']), 'issues': inspected['errors']}
    selected = select_frames(inspected['pairs'], max_frames, controls)
    report['sampling'] = {'requested_max_frames': max_frames, 'selected_frames': len(selected),
        'all_control_frames_included_even_above_limit': True, 'selected_ids': [x['id'] for x in selected]}
    by_id = defaultdict(list)
    for p in controls: by_id[p['id']].append(p)
    if not all(report['analytic_checks'].values()): report['issues'].append({'level': 'error', 'field': 'implementation', 'detail': 'Analytic contract failed'})
    try:
        if any(p['level'] == 'error' for p in report['issues']): raise ValueError('Static configuration errors prevent projection')
        geo = dict(c['geo']); geo['dsm'] = str(path(geo['dsm'])) if geo.get('dsm') else None
        if geo['normal_to_ellipsoid_m'] is None: raise ValueError('Normal-to-ellipsoid conversion not supplied')
        with ExitStack() as stack:
            satellite = stack.enter_context(GeoRaster(path(geo['satellite']), geo['authority']))
            ground = Ground(geo); stack.callback(ground.close)
            if not satellite.crs.is_projected or any(abs(a.unit_conversion_factor-1) > 1e-8 for a in satellite.crs.axis_info[:2]):
                raise ValueError('Satellite must use a metre-based projected CRS, as required by training')
            report['satellite'] = satellite.metadata()
            report['surface'] = ground.raster.metadata() if ground.raster else {'explicit_flat_plane_m': geo['flat_ground_normal_m'], 'warning': 'Not terrain validation'}
            for item in selected:
                try:
                    row = inspect_frame(item, c, satellite, ground, by_id[item['id']], position_source)
                    image_name = f"frame_{len(report['frames']):03}.png"
                    try: draw_overlay(item, row, c, satellite, out/image_name); row['overlay'] = image_name
                    except (ValueError, OSError) as e: row['warnings'].append('Overlay unavailable: '+str(e))
                except (ValueError, OSError, np.linalg.LinAlgError) as e:
                    row = {'id': item['id'], 'group': item['group'], 'errors': [str(e)], 'warnings': [],
                           'controls': [{**p, 'error': 'Frame projection failed: '+str(e)} for p in by_id[item['id']]]}
                report['frames'].append(row); report['controls'].extend(row['controls'])
    except (ValueError, OSError) as e:
        report['issues'].append({'level': 'missing_or_invalid', 'field': 'projection_inputs', 'detail': str(e)})
    done = {p['line'] for p in report['controls']}
    report['controls'] += [{**p, 'error': 'No evaluated image/geometry for this control id'} for p in controls if p['line'] not in done]
    report['control_metrics'] = {role: summarize_controls([p for p in report['controls'] if p['role'] == role], tolerance) for role in ('calibration', 'check')}
    report['control_metrics_by_group'] = {group: summarize_controls([p for p in report['controls'] if p['id'].split('/')[0] == group], tolerance) for group in sorted({p['id'].split('/')[0] for p in controls})}
    report['summary'] = {'matched_frames': len(inspected['pairs']), 'evaluated_frames': len(report['frames']),
        'frames_with_errors': sum(bool(r['errors']) for r in report['frames']),
        'frames_without_overlay': sum('overlay' not in r for r in report['frames']),
        'frames_with_full_footprint': sum('footprint_wgs84' in r for r in report['frames']),
        'independent_ground_controls_provided': bool(controls), 'real_geometry_validated': False}
    frame_rows = []
    for row in report['frames']:
        metrics = summarize_controls(row['controls'], tolerance)
        status = ('projection_failed' if row['errors'] else 'control_failed' if metrics['failed_or_unmatched'] else
                  'control_out_of_tolerance' if metrics['evaluated'] and metrics.get('all_within_tolerance') is False else
                  'control_checked' if metrics['evaluated'] and tolerance is not None else
                  'control_measured_no_tolerance' if metrics['evaluated'] else 'footprint_computable_unverified')
        row['diagnostic_status'] = status
        frame_rows.append({'id': row['id'], 'group': row['group'], 'status': status,
            'control_count': metrics['points'], 'control_evaluated': metrics['evaluated'],
            'max_control_error_m': metrics.get('max_m'), 'tolerance_m': tolerance,
            'errors': row['errors'], 'warnings': row['warnings'], 'overlay': row.get('overlay')})
    failed = (not report['frames'] or report['mapping']['issues'] or report['summary']['frames_with_errors']
              or any(x['level'] != 'warning' for x in report['issues']) or any('error' in x for x in report['controls'])
              or (tolerance is not None and any(x.get('error_m', 0) > tolerance for x in report['controls'])))
    report['diagnostic_status'] = 'failed' if failed else 'checks_passed_not_calibration_confirmation'
    write_csv(out/'frames.csv', ['id','group','status','control_count','control_evaluated','max_control_error_m','tolerance_m','errors','warnings','overlay'], frame_rows)
    write_csv(out/'controls.csv', ['id','line','role','pixel_x','pixel_y','longitude','latitude','error_m','error_east_m','error_north_m','error'], report['controls'])
    features = [{'type': 'Feature', 'properties': {'id': r['id'], 'kind': 'predicted_footprint', 'unverified': True},
                 'geometry': r['footprint_wgs84']} for r in report['frames'] if 'footprint_wgs84' in r]
    write_json(out/'footprints_wgs84.geojson', {'type': 'FeatureCollection', 'features': features})
    write_json(out/'geometry_report.json', report)
    rows = ''.join('<tr><td>'+html.escape(r['id'])+'</td><td>'+html.escape(r['diagnostic_status'])+'</td><td>'+html.escape('; '.join(r['errors']))+'</td></tr>' for r in report['frames'])
    pictures = ''.join(f'<h2>{html.escape(r["id"])}</h2><img src="{r["overlay"]}">' for r in report['frames'] if 'overlay' in r)
    page = '<!doctype html><meta charset="utf-8"><title>几何预检</title><style>body{font:16px sans-serif;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}td{padding:10px;border:1px solid #ccc}pre{white-space:pre-wrap}</style>'
    page += '<h1>几何预检：不会自动确认标定</h1><p>TXT 的平台位置真值不等于图像中心落点。请用独立图像—地面同名点确认几何约定。</p>'
    page += '<p><strong>状态：'+html.escape(report['diagnostic_status'])+'</strong> · '+('合成数据，不代表真实精度' if report['synthetic'] else '真实几何仍需独立标定确认')+'</p>'
    page += '<p><a href="geometry_report.json">完整报告</a> · <a href="mapping_report.json">图像/TXT配对</a> · <a href="footprints_wgs84.geojson">WGS84足迹</a></p>'
    page += '<p><a href="frames.csv">逐帧状态 CSV</a> · <a href="controls.csv">控制点误差 CSV</a></p><p>control_out_of_tolerance = 控制点超差；footprint_computable_unverified = 足迹可计算，但未做地标核验。</p>'
    import json
    page += '<h2>检查与缺项</h2><pre>'+html.escape(json.dumps({'summary': report['summary'], 'issues': report['issues'], 'control_metrics': report['control_metrics']}, ensure_ascii=False, indent=2))+'</pre><table>'+rows+'</table>'+pictures
    (out/'index.html').write_text(page, encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml')
    p.add_argument('--output', default='artifacts/geometry_validation'); p.add_argument('--max-frames', type=int, default=16)
    p.add_argument('--controls'); p.add_argument('--tolerance-m', type=float); p.add_argument('--position-source', choices=['gt', 'prior'], default='gt')
    a = p.parse_args(); report = run(config(a.config), a.output, a.max_frames, a.controls, a.tolerance_m, a.position_source)
    print(report['summary']); print('Diagnostic only; open', str(Path(a.output)/'index.html'))
    raise SystemExit(2 if report['diagnostic_status'] == 'failed' else 0)
