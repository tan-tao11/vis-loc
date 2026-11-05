import argparse
import platform
import shutil
import sys
from common import config, path, write_json
from records import scan
from geometry import GeoRaster

def inspect(c):
    report = scan(c)
    report['system'] = {'platform': platform.platform(), 'python': sys.version,
                        'free_output_bytes': shutil.disk_usage(path('.')).free}
    report['rasters'] = {}
    for name in ('satellite', 'dsm'):
        try:
            with GeoRaster(path(c['geo'][name]), c['geo']['authority' if name == 'satellite' else 'dsm_authority']) as r:
                report['rasters'][name] = r.metadata()
        except Exception as e:
            report['errors'].append({'reason': 'geo_metadata', 'raster': name, 'detail': str(e)})
    try:
        from prepare_dataset import satellite_intensity_check
        from scc_inference import reference_config, open_reference
        ref = reference_config(c, c['inference']['reference']['window_m'])
        report['rasters']['satellite_intensity'] = satellite_intensity_check(ref)
        r = open_reference(c, ref['window_m'])
        report['rasters']['inference_map_grid'] = {'crs': str(r.forward.target_crs), 'resolution_m': r.res, 'shape': list(r.shape)}
        r.close()
    except Exception as e:
        report['errors'].append({'reason': 'inference_reference', 'detail': str(e)})
    report['missing_confirmations'] = [f'{section}.{key}' for section, key in
        [('data', 'mapping_confirmed'), ('pose', 'calibration_confirmed'), ('geo', 'vertical_confirmed'),
         ('preprocess', 'deployment_confirmed'), ('model', 'deployed_backbone_confirmed')] if not c[section][key]]
    write_json(path(c['output'])/'inspection.json', report)
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); a = p.parse_args()
    r = inspect(config(a.config))
    print(f"Matched {len(r['pairs'])} images; {len(r['errors'])} issues. See inspection.json")
    sys.exit(1 if r['errors'] else 0)
