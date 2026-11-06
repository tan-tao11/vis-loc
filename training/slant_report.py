import sys, math
from shapely.geometry import shape
from common import read_json
from scc_inference import open_reference
m = read_json(sys.argv[1]); c = m['config']
ref = open_reference(c, 600.)
for g in m['groups']:
    r = g['record']; gx, gy = ref.forward.transform(r['droneLongitude_gt'], r['droneLatitude_gt'])
    d = [math.hypot(x-gx, y-gy) for x, y in shape(g['query_footprint']).exterior.coords]
    pos = sum(x['label'] == 1 for x in g['candidates'])
    print(f"{g['id']:28s} AGL {g['agl_m']:6.0f} elev {g['elevation_deg']:6.1f} block {g['side_m']:6.0f} window {g['window_m']:6.0f} "
          f"footprint {min(d):6.0f}-{max(d):6.0f} m from aircraft, positives {pos}/{len(g['candidates'])}")
for f in m['failures'][:20]: print('FAIL', f['id'], f['reason'])
ref.close()
