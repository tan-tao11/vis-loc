"""Online image access and legal grouped sampling; no frozen feature cache."""
import random
import numpy as np
import torch
from common import path, read_json, verify_manifest, require_real_ready, digest
from geometry import GeoRaster
from preprocess import query_image, candidate_image, tensor_image


def load_manifest(filename):
    m = read_json(filename)
    verify_manifest(m)
    require_real_ready(m['config'])
    if m['leakage']: raise ValueError('LoRA training refuses split leakage')
    if m['config']['preprocess']['size'] != 336: raise ValueError('LoRA expects deployed 336x336 inputs')
    return m


class OnlineImages:
    def __init__(self, manifest):
        self.manifest = manifest
        self.config = manifest['config']
        c = self.config
        self.satellite = GeoRaster(path(c['geo']['satellite']), c['geo']['authority'])

    def close(self): self.satellite.close()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def read(self, group, index=None):
        c = self.config
        if index is None: u, _ = query_image(group['image'], c['preprocess'])
        else: u, _ = candidate_image(self.satellite, group['candidates'][index], c['preprocess'], c['candidates']['max_window_pixels'])
        return tensor_image(u, c['preprocess'])


def encode_group(model, images, group, device, batch_size, amp, indices=None):
    """Works WITH gradients for training and under no_grad for evaluation/mining."""
    if batch_size < 1: raise ValueError('image_batch_size must be positive')
    selected = list(range(len(group['candidates']))) if indices is None else list(indices)
    query = images.read(group).unsqueeze(0).to(device)
    enabled = amp and torch.device(device).type == 'cuda'
    with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=enabled):
        q = model(query)[0]
    features = []
    for start in range(0, len(selected), batch_size):
        batch = torch.stack([images.read(group, i) for i in selected[start:start+batch_size]]).to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=enabled):
            features.append(model(batch))
    gallery = torch.cat(features) if features else q.new_empty((0, 1024))
    return q, gallery


def eligible_training_groups(manifest):
    return [g for g in manifest['groups'] if g['split'] == 'train' and
            any(c['label'] == 1 for c in g['candidates']) and any(c['label'] == 0 for c in g['candidates'])]


def balanced_schedule(groups, rng):
    buckets = {}
    for g in groups: buckets.setdefault(g['group'], []).append(g)
    if not buckets: raise ValueError('No train groups with both legal positives and negatives')
    maximum = max(map(len, buckets.values())); schedule = []
    for bucket in buckets.values():
        # Preserve every original training frame each epoch; oversample smaller groups only.
        schedule.extend(bucket + [rng.choice(bucket) for _ in range(maximum-len(bucket))])
    rng.shuffle(schedule)
    return schedule


def sample_candidates(group, hard_indices, tc, rng):
    if group['split'] != 'train': raise ValueError('Candidate subsampling is training-only')
    positives = [i for i, c in enumerate(group['candidates']) if c['label'] == 1]
    negatives = [i for i, c in enumerate(group['candidates']) if c['label'] == 0]
    if not positives or not negatives: raise ValueError('Training group needs positives and legal negatives')
    if any(i not in negatives for i in hard_indices): raise ValueError('Hard list includes positive, ignore or foreign candidate')
    selected = rng.sample(positives, min(tc['max_positives'], len(positives)))
    hard = list(dict.fromkeys(hard_indices))[:tc['hard_negatives']]
    remaining = [i for i in negatives if i not in hard]
    chosen_random = rng.sample(remaining, min(tc['random_negatives'], len(remaining)))
    selected += hard + chosen_random
    if not hard and not chosen_random: selected.append(rng.choice(negatives))
    return selected
