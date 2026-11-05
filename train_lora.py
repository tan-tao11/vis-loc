"""Single-GPU online LoRA training. Shared adapter, grouped contrastive objective."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import random
import numpy as np
import torch
from common import config, seed_all, write_json, digest, sha256, ROOT
from lora import build_lora
from lora_data import load_manifest, OnlineImages, encode_group, eligible_training_groups, balanced_schedule, sample_candidates
from losses import grouped_loss
from evaluate_lora import evaluate_online
from outputs import fresh_directory, write_training_history


def validate_settings(c):
    tc = c['train']
    for key in ('epochs', 'image_batch_size', 'accumulation_groups', 'max_positives', 'hard_refresh_epochs', 'patience'):
        if not isinstance(tc[key], int) or isinstance(tc[key], bool) or tc[key] <= 0:
            raise ValueError(key+' must be a positive integer')
    for key in ('hard_negatives', 'random_negatives'):
        if not isinstance(tc[key], int) or tc[key] < 0: raise ValueError('Invalid '+key)
    if tc['hard_negatives']+tc['random_negatives'] < 1: raise ValueError('At least one negative required')
    for key in ('lr', 'temperature', 'clip_grad_norm'):
        if not math.isfinite(tc[key]) or tc[key] <= 0: raise ValueError('Invalid '+key)
    if tc['weight_decay'] < 0 or tc['min_delta'] < 0: raise ValueError('Invalid decay/min_delta')
    if c['evaluation']['image_batch_size'] < 1: raise ValueError('Evaluation batch size must be positive')


def state_fingerprint(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def code_identity():
    return {name: sha256(ROOT/name) for name in ('lora.py', 'lora_data.py', 'train_lora.py', 'evaluate_lora.py', 'losses.py')}


@torch.no_grad()
def mine_online(model, images, groups, tc, device):
    previous = model.training; model.eval(); result = {}
    try:
        for g in groups:
            if g['split'] != 'train': raise ValueError('LoRA hard-negative mining is restricted to train')
            q, gallery = encode_group(model, images, g, device, tc['image_batch_size'], tc['amp'])
            scores = (gallery @ q).cpu().numpy()
            negatives = [i for i, cand in enumerate(g['candidates']) if cand['label'] == 0]
            appearance = sorted(negatives, key=lambda i: -float(scores[i]))
            center = np.array(g['ground_center_map'])
            near = sorted(negatives, key=lambda i: np.linalg.norm(np.array(g['candidates'][i]['center_map'])-center))
            # Reserve a quarter of mined slots for geographically close valid negatives.
            quota = min(tc['hard_negatives'], max(1, tc['hard_negatives']//4))
            selected = near[:quota]
            selected += [i for i in appearance if i not in selected][:max(0, tc['hard_negatives']-len(selected))]
            result[g['id']] = selected
    finally:
        model.train(previous)
    return result


def atomic_save(filename, payload):
    filename = Path(filename); temporary = filename.with_suffix(filename.suffix+'.partial')
    torch.save(payload, temporary); os.replace(temporary, filename)


def train_lora(c, manifest_file, output, device=None, resume=None):
    validate_settings(c); seed_all(c['seed'])
    m = load_manifest(manifest_file); device = device or c['train']['device']; tc = c['train']
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; explicitly choose --device cpu for diagnostics')
    torch.backends.cuda.matmul.allow_tf32 = bool(tc['allow_tf32'])
    torch.backends.cudnn.allow_tf32 = bool(tc['allow_tf32'])
    groups = eligible_training_groups(m)
    if not groups or not any(g['split'] == 'val' for g in m['groups']):
        raise ValueError('Separate validation geography and train groups with positive/negative candidates required')
    out = Path(output)
    if not resume: fresh_directory(out)
    model, identity = build_lora(m['config'], c['lora'], device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=tc['lr'], weight_decay=tc['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc['epochs'])
    scaler = torch.amp.GradScaler('cuda', enabled=tc['amp'] and torch.device(device).type == 'cuda', init_scale=1024.)
    rng = random.Random(c['seed']); history = []; first_epoch = 0; best_key = None; stale = 0; hard = {}
    run_identity = {'manifest_id': m['manifest_id'], 'settings': digest(c), 'backbone': identity, 'code': code_identity()}
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['run_identity'] != run_identity: raise ValueError('Resume requires identical data/config/model/code')
        model.load_adapter_state(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        history, first_epoch, best_key, stale = saved['history'], saved['epoch']+1, saved['best_key'], saved['stale']
        hard = saved['hard']; rng.setstate(saved['rng_python']); torch.set_rng_state(saved['rng_torch'])
        if torch.device(device).type == 'cuda': torch.cuda.set_rng_state(saved['rng_cuda'], device)
        if not (out/'best.pth').exists(): raise ValueError('Resume in the original run directory retaining best.pth')
    write_json(out/'run.json', {'method': 'lora_shared_qv_gem1024', 'run_identity': run_identity,
        'settings': c, 'trainable_parameters': model.assert_freeze(), 'injected_blocks': model.block_indices,
        'training_groups': len(groups), 'train_groups_excluded_no_positive_or_negative':
            sum(g['split'] == 'train' for g in m['groups'])-len(groups),
        'online_pixels': True, 'cached_backbone_features': False, 'synthetic': m['synthetic']})
    with OnlineImages(m) as images:
        for epoch in range(first_epoch, tc['epochs']):
            if not hard or epoch % tc['hard_refresh_epochs'] == 0:
                hard = mine_online(model, images, groups, tc, device)
                write_json(out/f'hard_negatives_epoch_{epoch:03}.json', {
                    'split': 'train', 'manifest_id': m['manifest_id'],
                    'adapter_fingerprint': state_fingerprint(model.adapter_state()), 'indices': hard})
            schedule = balanced_schedule(groups, rng); model.train(); losses = []; updates = 0; overflows = 0
            gradients = {'A': 0., 'B': 0.}
            if torch.device(device).type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            for offset in range(0, len(schedule), tc['accumulation_groups']):
                window = schedule[offset:offset+tc['accumulation_groups']]
                optimizer.zero_grad(set_to_none=True)
                try:
                    for g in window:
                        selected = sample_candidates(g, hard[g['id']], tc, rng)
                        q, gallery = encode_group(model, images, g, device, tc['image_batch_size'], tc['amp'], selected)
                        labels = torch.tensor([g['candidates'][i]['label'] for i in selected], device=device)
                        loss, valid = grouped_loss(q, gallery, labels, tc['temperature'])
                        if valid != 1 or not torch.isfinite(loss): raise ValueError('Invalid training loss')
                        scaler.scale(loss/len(window)).backward()
                        losses.append(float(loss.detach()))
                        del q, gallery, loss
                    scaler.unscale_(optimizer)
                    model.assert_freeze()
                    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
                    if finite:
                        for name, p in model.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                gradients['A' if '.lora_A.' in name else 'B'] += float(p.grad.norm())
                        torch.nn.utils.clip_grad_norm_(parameters, tc['clip_grad_norm'])
                        updates += 1
                    else:
                        if not scaler.is_enabled(): raise ValueError('Nonfinite LoRA gradients')
                        overflows += 1
                    scaler.step(optimizer); scaler.update()
                except torch.cuda.OutOfMemoryError as e:
                    raise RuntimeError('LoRA CUDA OOM: lower image_batch_size/positive/negative counts, retain checkpointing. No partial batch was accepted.') from e
            if updates == 0: raise RuntimeError('No successful optimizer update; inspect AMP overflow or labels')
            scheduler.step(); model.assert_freeze()
            report = evaluate_online(model, images, m, 'val', device, c['evaluation']['image_batch_size'], tc['amp'], compare_baseline=False)
            metrics = report['results']['lora_gem1024']['overall']
            recall = metrics['recall_all_legal']; rank = metrics['mean_first_positive_rank_covered']
            # Validation only: R@1 first, then R@5 and earliest positive rank break ties.
            key = [recall['1'], recall['5'], -rank if rank is not None else -1e30]
            row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'val_recall_all': recall,
                   'gradient_norm_sum': gradients, 'optimizer_updates': updates, 'amp_overflow_updates_skipped': overflows,
                   'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else 0}
            history.append(row); print(row, flush=True)
            improved = best_key is None or key[0] > best_key[0]+tc['min_delta'] or (key[0] == best_key[0] and key[1:] > best_key[1:])
            state = model.adapter_state()
            metadata = {'kind': 'scc_dinov2_lora_training', 'format_version': 1, 'lora': c['lora'],
                'state_dict': state, 'backbone': identity, 'preprocess': m['config']['preprocess'], 'pose': m['config']['pose'],
                'gem': {'p': 4., 'clamp_min': 1e-6, 'dimension': 1024}, 'output_dim': 1024,
                'manifest_id': m['manifest_id'], 'synthetic_only': m['synthetic'], 'selection_split': 'val',
                'epoch': epoch, 'val_selection_key': key, 'run_identity': run_identity}
            if improved:
                best_key = key; stale = 0; atomic_save(out/'best.pth', metadata); write_json(out/'best_validation.json', report)
            else: stale += 1
            atomic_save(out/'last.pth', {**metadata, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(), 'history': history, 'best_key': best_key, 'stale': stale, 'hard': hard,
                'rng_python': rng.getstate(), 'rng_torch': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None})
            write_training_history(out, history)
            if stale >= tc['patience']: break
    return out/'best.pth'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', default='configs/lora.yaml')
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True); p.add_argument('--device'); p.add_argument('--resume')
    a = p.parse_args(); train_lora(config(a.config), a.manifest, a.output, a.device, a.resume)
