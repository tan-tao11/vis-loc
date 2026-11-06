import sys, json, numpy as np, torch
from common import read_json
from cache import group_cache_name
from evaluate import load_heads_checkpoint, rank_metrics
manifest, cache, ckpt, key = sys.argv[1:5]
m = read_json(manifest); heads, _ = load_heads_checkpoint(ckpt, 'cpu')
for g in [g for g in m['groups'] if key in g['id'] or key in str(g.get('image'))]:
    with np.load(f'{cache}/{group_cache_name(g["id"])}', allow_pickle=False) as f: q, s = torch.from_numpy(f['query']), torch.from_numpy(f['satellite'])
    labels = [x['label'] for x in g['candidates']]
    with torch.no_grad(): r = {'gem1024': rank_metrics((s @ q).numpy(), labels), 'heads': rank_metrics((heads(s, 'satellite') @ heads(q, 'ir')).numpy(), labels)}
    print(g['id'], g['split'], 'candidates', len(labels), 'positives', labels.count(1),
          {k: {'first_positive_rank': v['first_positive_rank'], 'top5': v['order'][:5]} for k, v in r.items()},
          'positive_idx', [i for i, x in enumerate(labels) if x == 1])
