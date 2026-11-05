import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from common import read_json, write_json
from cache import load_cache
from models import RetrievalHeads

def rank_metrics(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape: raise ValueError('Scores/labels must be equal-length vectors')
    if not np.isin(labels, [-1, 0, 1]).all(): raise ValueError('Invalid candidate label')
    if not np.isfinite(scores).all(): raise ValueError('Invalid scores')
    order = np.argsort(-scores, kind='stable')
    hits = np.flatnonzero(labels[order] == 1)
    return {'first_positive_rank': int(hits[0])+1 if hits.size else None,
            'covered': bool((labels == 1).any()), 'candidate_count': len(labels), 'order': order.tolist()}

def summarize(rows):
    n = len(rows); covered = sum(r['covered'] for r in rows)
    return {'queries': n, 'covered_queries': covered, 'candidate_coverage': covered/n if n else None,
            'mean_candidates': float(np.mean([r['candidate_count'] for r in rows])) if n else None,
            'mean_first_positive_rank_covered': float(np.mean([r['first_positive_rank'] for r in rows if r['covered']])) if covered else None,
            'recall_all_legal': {str(k): sum(r['first_positive_rank'] is not None and r['first_positive_rank'] <= k for r in rows)/n if n else None for k in (1, 5, 10, 20)},
            'recall_conditional_covered': {str(k): sum(r['first_positive_rank'] is not None and r['first_positive_rank'] <= k for r in rows)/covered if covered else None for k in (1, 5, 10, 20)}}

@torch.no_grad()
def evaluate_arrays(m, arrays, heads=None, split='val', device='cpu'):
    device = str(device)
    if not any(g['split'] == split for g in m['groups']): raise ValueError('Empty evaluation split: '+split)
    if heads: heads.eval()
    result = {}; collected = {}
    for mode in (['gem1024', 'heads'] if heads else ['gem1024']):
        rows = []; buckets = defaultdict(list); start = time.perf_counter()
        for g in m['groups']:
            if g['split'] != split: continue
            q, s = arrays[g['id']]; q = torch.from_numpy(q).to(device); s = torch.from_numpy(s).to(device)
            if mode == 'heads': q, s = heads(q, 'ir'), heads(s, 'satellite')
            scores = (s @ q).cpu().numpy()
            labels = [x['label'] for x in g['candidates']]
            row = {'id': g['id'], **rank_metrics(scores, labels)}; rows.append(row)
            for key in ('group', 'sequence', 'protocol'): buckets[key+':'+g[key]].append(row)
            # For each scale, keep every legal query, including zero candidates at that scale.
            collected.setdefault('scales', set()).update(x['scale_m'] for x in g['candidates'])
            row['_scores'] = scores.tolist(); row['_labels'] = labels
        if device.startswith('cuda'): torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        groups = {g['id']: g for g in m['groups']}
        for scale in sorted(collected.get('scales', [])):
            for row in rows:
                indexes = [i for i, x in enumerate(groups[row['id']]['candidates']) if x['scale_m'] == scale]
                buckets[f'scale_m:{scale}'].append(rank_metrics(np.array(row['_scores'])[indexes], np.array(row['_labels'])[indexes]))
        for row in rows:
            row['scores'] = row.pop('_scores'); del row['_labels']
        result[mode] = {'overall': summarize(rows), 'groups': {k: summarize(v) for k, v in buckets.items()},
                        'rows': rows, 'ranking_seconds': seconds, 'timing_scope': 'cached feature projection+cosine+rank; excludes backbone and I/O'}
    return {'split': split, 'synthetic': m['synthetic'], 'manifest_id': m['manifest_id'],
            'data_flow': {'matched': m['total_matched'], 'processed': m['processed'], 'legal': len(m['groups']),
                          'preparation_failures': m['failures'], 'limited': m['limited']}, 'results': result}

def load_heads_checkpoint(filename, device='cpu'):
    ckpt = torch.load(filename, map_location='cpu', weights_only=True)
    heads = RetrievalHeads(**ckpt['structure']); heads.load_state_dict(ckpt['state_dict'], strict=True)
    return heads.to(device).eval(), ckpt

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', required=True); p.add_argument('--cache', required=True)
    p.add_argument('--checkpoint'); p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--device', default='cpu'); p.add_argument('--output', required=True); a = p.parse_args()
    from outputs import write_evaluation, evaluation_attempt
    with evaluation_attempt(a.output):
        m, arrays, index = load_cache(a.manifest, a.cache)
        heads, checkpoint = load_heads_checkpoint(a.checkpoint, a.device) if a.checkpoint else (None, None)
        if checkpoint and checkpoint['cache_fingerprint'] != index['fingerprint']: raise ValueError('Checkpoint/candidate cache mismatch')
        result = evaluate_arrays(m, arrays, heads, a.split, a.device); result['extraction'] = index
        print('Report:', write_evaluation(a.output, result, m, render_images=True)/'index.html')
    print({k: v['overall'] for k, v in result['results'].items()})
