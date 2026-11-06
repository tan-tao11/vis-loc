import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from common import config, path, read_json, write_json
from geometry import camera_rotation, intrinsics, CameraFrame, Ground, intersect
from preprocess import query_image
from scc_inference import TileSource, reference_config, block_bounds, map_to_rotated
from shapely.geometry import shape
from outputs import fresh_directory


def evaluation_orders(m, report, mode=None):
    if report['manifest_id'] != m['manifest_id'] or report['synthetic'] != m['synthetic']:
        raise ValueError('Evaluation/manifest mismatch; refuse misleading overlay')
    mode = mode or next((x for x in ('lora_gem1024', 'mlp512', 'heads', 'gem1024') if x in report['results']), None)
    if mode not in report['results']: raise ValueError('Unknown evaluation mode: '+str(mode))
    groups = {g['id']: g for g in m['groups'] if g['split'] == report['split']}
    rows = report['results'][mode]['rows']
    if len(rows) != len(groups) or {r['id'] for r in rows} != set(groups):
        raise ValueError('Evaluation rows are incomplete or duplicated')
    for row in rows:
        count = len(groups[row['id']]['candidates'])
        if row['candidate_count'] != count or sorted(row['order']) != list(range(count)):
            raise ValueError('Candidate order/count mismatch: '+row['id'])
        if 'scores' in row and (np.asarray(row['scores']).shape != (count,) or not np.isfinite(row['scores']).all()):
            raise ValueError('Invalid visualization scores: '+row['id'])
    return mode, {r['id']: r for r in rows}

def bice_examples(c, output):
    rows = [dict() for _ in range(4)]
    for line in path('reference/bice.md').read_text(encoding='utf-8').splitlines():
        cells = [v.strip() for v in line.strip().strip('|').split('|')]
        if len(cells) != 5: continue
        try: values = list(map(float, cells[1:]))
        except ValueError: continue
        for r, v in zip(rows, values): r[cells[0]] = v
    geo = dict(c['geo'], dsm=None, flat_fallback=True, flat_ground_normal_m=1500., normal_to_ellipsoid_m=0.)
    ground = Ground(geo); result = []
    for group, r in zip(('DS', 'DV', 'NS', 'NV'), rows):
        R = camera_rotation(r, c['pose']); K = intrinsics(640, 512, r)
        frame = CameraFrame(r['droneLongitude_gt'], r['droneLatitude_gt'], r['droneAltitude_gt'], 0.)
        entry = {'group': group, 'record': r, 'rotation': R.tolist(), 'optical_ned': R[:, 2].tolist(),
                 'camera_right_ned': R[:, 0].tolist(), 'camera_down_ned': R[:, 1].tolist(),
                 'synthetic_plane_normal_height_m': 1500., 'warning': 'Synthetic surface, candidate axes; NOT calibrated real map output'}
        try:
            pixels = [(320, 256)]+[(x, y) for x, y in [(0, 0), (640, 0), (640, 512), (0, 512)]]
            hits = [intersect(frame, R@np.linalg.inv(K)@np.array([x, y, 1]), ground, 10, 30000).tolist() for x, y in pixels]
            entry.update(center_lon_lat_normal=hits[0], footprint_lon_lat_normal=hits[1:])
            entry['right_heading_true_north_deg'] = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))
        except ValueError as e: entry['rejected'] = str(e)
        result.append(entry)
    write_json(output, result)
    return result

def contact_sheet(manifest, output, evaluation=None, mode=None):
    m = manifest if isinstance(manifest, dict) else read_json(manifest)
    c = m['config']; rows_by_id = {}; errors = []
    if evaluation:
        report = evaluation if isinstance(evaluation, dict) else read_json(evaluation)
        mode, rows_by_id = evaluation_orders(m, report, mode)
    selected = []
    for group in ('DS', 'DV', 'NS', 'NV'):
        options = [g for g in m['groups'] if g['group'] == group and (not evaluation or g['id'] in rows_by_id)]
        if evaluation:
            options.sort(key=lambda g: (-(rows_by_id[g['id']]['first_positive_rank'] or 10**9), g['id']))
        if options: selected.append(options[0])
    if not selected: raise ValueError('No queries to visualize')
    canvas = Image.new('RGB', (1450, 290*max(1, len(selected))+50), '#101924'); draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), 'SYNTHETIC DIAGNOSTICS - NOT REAL ACCURACY' if m['synthetic'] else 'CALIBRATION / RETRIEVAL DIAGNOSTICS', fill='white')
    draw.text((650, 12), ('mode: '+mode+' | worst query per group in '+report['split']) if evaluation else 'Candidate preview (NOT ranking; manifest order)', fill='#ffcb55')
    tiles = TileSource(c)
    try:
        from prepare_dataset import satellite_intensity_check
        try:
            stats = satellite_intensity_check(reference_config(c, c['inference']['reference']['window_m']))
            draw.text((15, 30), f"satellite {stats['dtypes']} bands {stats['image_bands']} range {stats['image_value_range']} "
                      f"raw p0.5/p50/p99.5 {[round(v, 2) for v in stats['raw_percentiles'].values()]} "
                      f"white {stats['mapped_white_fraction']:.2f} black {stats['mapped_black_fraction']:.2f}", fill='#a0b0c4')
        except ValueError as e:
            errors.append({'panel': 'satellite_intensity', 'error': str(e)})
            draw.text((15, 30), 'SATELLITE VALUE RANGE WRONG (tiles saturate); see contact_sheet.png.json', fill='#ff7777')
        _draw_rows(m, c, selected, rows_by_id, evaluation, tiles, canvas, draw, errors)
    finally: tiles.close()
    Path(output).parent.mkdir(parents=True, exist_ok=True); canvas.save(output)
    status = {'status': 'partial' if errors else 'complete', 'mode': mode, 'manifest_id': m['manifest_id'],
        'synthetic': m['synthetic'], 'selected_ids': [g['id'] for g in selected], 'errors': errors,
        'ordering': 'model score; worst query per group' if evaluation else 'manifest order; NOT retrieval ranking'}
    write_json(str(output)+'.json', status)
    return status


def _draw_rows(m, c, selected, rows_by_id, evaluation, tiles, canvas, draw, errors):
    """Panels show exactly what the deployed retrieval scores (pre-SGVA), built by scc_inference."""
    import cv2
    for i, g in enumerate(selected):
        top = 55+i*290
        try:
            q = tiles.query(g, c['preprocess']['size'])
            canvas.paste(Image.fromarray(q).convert('RGB').resize((210, 210)), (15, top))
        except (ValueError, OSError) as e:
            errors.append({'id': g['id'], 'panel': 'query', 'error': str(e)})
            draw.text((15, top+60), 'Query render failed; see JSON', fill='#ff7777')
        draw.text((15, top+214), g['id'], fill='white')
        cs = g['candidates']; order = rows_by_id[g['id']]['order'] if evaluation else list(range(len(cs)))
        try: frame = tiles.frame(g)
        except Exception as e:
            errors.append({'id': g['id'], 'panel': 'search_map', 'error': str(e)})
            draw.text((240, top+70), 'Search map failed; see JSON', fill='#ff7777'); continue
        footprint = np.asarray(shape(g['query_footprint']).exterior.coords)
        fp_rot = map_to_rotated(frame, footprint)
        # Overview: rotated search map as fed to retrieval; grey = all scored blocks, yellow = shown top-3, green = footprint.
        ref = frame['ref_image']; rgb = ref if ref.ndim == 3 else np.repeat(ref[..., None], 3, -1)
        overview = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); od = ImageDraw.Draw(overview)
        for k, cand in enumerate(cs):
            r0, r1, c0, c1 = block_bounds(cand['mid'], frame['side_px'])
            od.rectangle((c0, r0, c1, r1), outline=(255, 203, 85) if k in order[:3] else (120, 130, 140), width=max(1, ref.shape[0]//200))
        od.line([tuple(p) for p in fp_rot], fill=(46, 255, 160), width=max(2, ref.shape[0]//150))
        canvas.paste(overview.resize((210, 210)), (240, top))
        draw.text((240, top+214), f"search map {g['window_m']:.0f} m, {len(cs)}/{g['grid_size']} blocks", fill='white')
        for k, j in enumerate(order[:3]):
            x = 465+k*225; cand = cs[j]
            try: img = tiles.tile(g, j, c['preprocess']['size'])
            except Exception as e:
                errors.append({'id': g['id'], 'panel': 'candidate', 'candidate_index': j, 'error': str(e)})
                draw.text((x, top+60), f'Candidate {j} failed; see JSON', fill='#ff7777'); continue
            rgb = img if img.ndim == 3 else np.repeat(img[..., None], 3, -1)
            tile = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).resize((210, 210)); td = ImageDraw.Draw(tile)
            r0, r1, c0, c1 = block_bounds(cand['mid'], frame['side_px'])
            td.line([((p[0]-c0)/(c1-c0)*210, (p[1]-r0)/(r1-r0)*210) for p in fp_rot], fill='#2effa0', width=2)
            canvas.paste(tile, (x, top))
            white = float((img == 255).all(-1).mean()) if img.ndim == 3 else float((img == 255).mean())
            prefix = f'rank {k+1}' if evaluation else f'preview {j}'
            draw.text((x, top+214), f"{prefix} label {cand['label']} cov {cand['coverage']:.2f}", fill='white')
            draw.text((x, top+231), f"{cand['scale_m']:.1f} m / valid {cand['valid_fraction']:.3f} / white {white:.2f}",
                      fill='#ff7777' if white > .9 else '#a0b0c4')
            if evaluation and 'scores' in rows_by_id[g['id']]:
                draw.text((x, top+247), f"index {j} / score {rows_by_id[g['id']]['scores'][j]:.6f}", fill='#ffcb55')
        r = g['record']; info = [g['group'], f"yaw {g['yaw_deg']:.2f} elev {g['elevation_deg']:.2f}", f"AGL {g['agl_m']:.1f} m",
             f"isEO={r['isEO']} slant={g['is_slant']}", f"INS {r['innsYaw']:.2f}, {r['innsPitch']:.2f}, {r['innsRoll']:.2f}",
             f"sensor {r['sensorYaw']:.2f}, {r['sensorPitch']:.2f}, {r['sensorRoll']:.2f}",
             f"block {g['side_m']:.1f} m = {g['side_px']} px", 'green: GT footprint', 'yellow: shown blocks', 'tiles = deployed input, pre-SGVA']
        for k, line in enumerate(info): draw.text((1150, top+k*20), line, fill='white')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--manifest'); p.add_argument('--evaluation')
    p.add_argument('--mode', help='gem1024, heads, mlp512 or lora_gem1024; default prefers adapted model')
    p.add_argument('--output', default='artifacts/diagnostics'); a = p.parse_args(); out = fresh_directory(a.output)
    bice_examples(config(a.config), out/'bice_synthetic_plane.json')
    if a.manifest:
        status = contact_sheet(a.manifest, out/'contact_sheet.png', a.evaluation, a.mode)
        print('Contact sheet:', out/'contact_sheet.png', status['status'])
        raise SystemExit(2 if status['errors'] else 0)
