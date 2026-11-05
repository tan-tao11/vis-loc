import argparse
import copy
import os
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from common import config, write_json, seed_all, digest
from models import create_heads
from losses import grouped_loss
from cache import load_cache
from evaluate import evaluate_arrays
from outputs import fresh_directory, write_training_history

@torch.no_grad()
def mine_hard_negatives(heads, groups, arrays, device, count):
    heads.eval(); result = {}
    for g in groups:
        if g['split'] != 'train': raise ValueError('Mining is forbidden outside training split')
        q, s = arrays[g['id']]
        logits = (heads(torch.from_numpy(s).to(device), 'satellite') @ heads(torch.from_numpy(q).to(device), 'ir')).cpu().numpy()
        neg = [i for i, x in enumerate(g['candidates']) if x['label'] == 0]
        ranked = sorted(neg, key=lambda i: -float(logits[i]))
        # Include geographic near-but-nonoverlapping negatives as well as appearance confusers.
        center = np.array(g['ground_center_map'])
        near = sorted(neg, key=lambda i: np.linalg.norm(np.array(g['candidates'][i]['center_map'])-center))
        selected = sorted(set(ranked[:count]+near[:max(1, count//4)]))
        result[g['id']] = [i for i, x in enumerate(g['candidates']) if x['label'] == 1]+selected
    return result

def train(c, manifest, cache_dir, output, device=None):
    seed_all(c['seed']); device = device or c['train']['device']; tc = c['train']
    m, arrays, index = load_cache(manifest, cache_dir)
    if digest(c) != m['fingerprint']['config']: raise ValueError('Config/cache mismatch; create new run')
    if tc['loss'] != 'grouped_multi_positive': raise ValueError('Only one-direction grouped loss implemented')
    if tc['scheduler'] != 'cosine': raise ValueError('Unsupported scheduler')
    groups = [g for g in m['groups'] if g['split'] == 'train' and any(x['label'] == 1 for x in g['candidates'])]
    if not groups or not any(g['split'] == 'val' for g in m['groups']): raise ValueError('Training positives and separate geographic validation are required')
    out = fresh_directory(output)
    write_json(out/'run.json', {'manifest_id': m['manifest_id'], 'cache_fingerprint': index['fingerprint'],
        'synthetic': m['synthetic'], 'method': 'mlp512', 'selection_split': 'val'})
    heads = create_heads(c).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tc['epochs'])
    buckets = defaultdict(list)
    for g in groups: buckets[g['group']].append(g)
    history = []; best = -1.; stale = 0; hard = {}
    for epoch in range(tc['epochs']):
        if epoch % tc['hard_refresh_epochs'] == 0:
            hard = mine_hard_negatives(heads, groups, arrays, device, tc['hard_negative_count'])
            write_json(out/f'hard_negatives_epoch_{epoch:03}.json', {'split': 'train', 'cache_fingerprint': index['fingerprint'], 'selected_indices': hard})
        # Equal group frequency; use replacement only within train groups.
        schedule = []
        for _ in range(max(len(v) for v in buckets.values())):
            schedule.extend(random.choice(v) for v in buckets.values())
        random.shuffle(schedule); heads.train(); losses = []; gradients = {'ir': 0., 'satellite': 0.}
        for start in range(0, len(schedule), tc['batch_size']):
            optimizer.zero_grad(set_to_none=True); terms = []
            for g in schedule[start:start+tc['batch_size']]:
                q, s = arrays[g['id']]; selected = hard[g['id']]
                if not selected: continue
                q = heads(torch.from_numpy(q).to(device), 'ir')
                s = heads(torch.from_numpy(s[selected]).to(device), 'satellite')
                labels = torch.tensor([g['candidates'][j]['label'] for j in selected], device=device)
                loss, count = grouped_loss(q, s, labels, tc['temperature'])
                if count: terms.append(loss)
            if not terms: continue
            loss = torch.stack(terms).mean(); loss.backward()
            for domain in gradients:
                gradients[domain] += sum(float(p.grad.norm()) for p in getattr(heads, domain).parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step()
        report = evaluate_arrays(m, arrays, heads, 'val', device)
        score = report['results']['heads']['overall']['recall_all_legal']['1']
        if score is None: raise ValueError('Empty validation set')
        row = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None, 'val_recall_at_1_all': score, 'gradient_norm_sum': gradients}
        history.append(row); print(row, flush=True)
        if score > best:
            best = score; stale = 0
            ckpt = {'format_version': 1, 'state_dict': {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
                    'structure': heads.structure, 'config': c, 'epoch': epoch, 'val_recall_at_1_all': score,
                    'manifest_id': m['manifest_id'], 'cache_fingerprint': index['fingerprint'], 'backbone': index['backbone'],
                    'synthetic': m['synthetic'], 'selection_split': 'val'}
            torch.save(ckpt, out/'best.pth.partial'); os.replace(out/'best.pth.partial', out/'best.pth')
            write_json(out/'best_validation.json', report)
        else: stale += 1
        write_training_history(out, history)
        if stale >= tc['patience']: break
    return out/'best.pth'

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/train.yaml'); p.add_argument('--manifest', required=True)
    p.add_argument('--cache', required=True); p.add_argument('--output', required=True); p.add_argument('--device'); a = p.parse_args()
    train(config(a.config), a.manifest, a.cache, a.output, a.device)
