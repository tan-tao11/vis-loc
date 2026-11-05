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
    count = len(report['pairs']); maximum = c['candidates']['max_per_query']
    report['disk_estimate_upper_bound'] = {
        'queries': count, 'max_candidates_per_query': maximum,
        'float32_features_bytes': count*(maximum+1)*1024*4,
        'uncompressed_336_gray_patches_bytes_if_materialized': count*maximum*336*336,
        'patch_policy': 'No persistent patch copies; bounded window reads on demand'}
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
