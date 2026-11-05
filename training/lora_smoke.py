"""Real local ViT-L/14 LoRA forward/backward/export test on synthetic GIS images."""
import argparse
from pathlib import Path
import json
import torch
from common import config, write_json, read_json, sha256, ROOT, seed_all
from synthetic import make_fixture
from prepare_dataset import prepare
from offline_check import deny_network
from models import forward_frozen
from lora import build_lora
from lora_data import load_manifest, OnlineImages
from train_lora import train_lora
from evaluate_lora import evaluate_online
from export_lora import export_lora, load_lora_for_manifest
from integration.lora_adapter import LoRAInferenceAdapter


def run(output, device):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {'synthetic_only': True, 'device': device, 'network_scope': 'Python socket guard; PROJ_NETWORK=OFF', 'checks': {}}
    try:
        with deny_network():
            cfgfile = make_fixture(out/'data_fixture'); c = config(cfgfile)
            m = prepare(c); manifest = Path(c['output'])/'manifest.json'
            lc = config(ROOT/'configs/lora.yaml'); lc['train'].update(epochs=3, patience=3, max_positives=1,
                hard_negatives=2, random_negatives=1, accumulation_groups=2)
            lc['evaluation']['image_batch_size'] = 2
            import yaml
            (out/'smoke_lora.yaml').write_text(yaml.safe_dump(lc, sort_keys=False), encoding='utf-8')
            seed_all(lc['seed'])
            model, identity = build_lora(c, lc['lora'], device)
            with OnlineImages(m) as images: example = images.read(m['groups'][0]).unsqueeze(0).to(device)
            with torch.no_grad():
                baseline = forward_frozen(model.backbone, example, amp=False)
                zero = model(example)
                zero_error = float((baseline-zero).abs().max())
                if zero_error > 1e-6: raise AssertionError('Zero-init LoRA differs from original DINO')
            report['checks']['zero_init_max_abs_error'] = zero_error
            report['trainable_parameters'] = model.assert_freeze(); report['backbone'] = identity
            del model, baseline, zero
            if torch.device(device).type == 'cuda': torch.cuda.empty_cache()
            best = train_lora(lc, manifest, out/'training', device)
            payload = export_lora(best, out/'synthetic_ONLY_scc_retrieval_lora.pth')
            model, checkpoint = load_lora_for_manifest(best, m, device)
            with OnlineImages(m) as images:
                val = evaluate_online(model, images, m, 'val', device, 2, True, compare_baseline=True)
                test = evaluate_online(model, images, m, 'test', device, 2, True, compare_baseline=True)
            write_json(out/'validation.json', val); write_json(out/'test_final.json', test)
            adapter = LoRAInferenceAdapter(out/'synthetic_ONLY_scc_retrieval_lora.pth', c, True, device, allow_synthetic=True)
            with torch.no_grad():
                expected = model(example); actual = adapter.retrieve(example, amp=False)
                reload_error = float((expected-actual).abs().max())
                if reload_error > 1e-6: raise AssertionError('LoRA reload numerical mismatch')
                with model.without_lora(): original = model(example)
                difference = float((expected-original).abs().max())
                if difference == 0: raise AssertionError('Adapter had no effect after training')
                dense = torch.randn(1, 576, 1024, device=device); cls = torch.randn(1, 1024, device=device)
                _, retained_dense, retained_cls = adapter.replace_global((original, dense, cls), example, amp=False)
                assert retained_dense is dense and retained_cls is cls
                disabled = LoRAInferenceAdapter(enabled=False)
                assert disabled.retrieve(example, original) is original
            history = read_json(out/'training/history.json')
            assert any(h['gradient_norm_sum']['A'] > 0 and h['gradient_norm_sum']['B'] > 0 for h in history)
            report['checks'].update({'reload_max_abs_error': reload_error, 'trained_vs_original_max_abs_difference': difference,
                'SGVA_tensor_identity_preserved': True, 'disabled_original_identity_preserved': True,
                'base_frozen_gradient_assertions_passed': True, 'both_A_B_nonzero_gradients': True,
                'full_candidate_validation_and_test': True, 'no_frozen_feature_cache': True})
            report['loss_history'] = [h['loss'] for h in history]
            report['peak_cuda_allocated_bytes'] = max(h['peak_cuda_allocated_bytes'] for h in history)
            report['adapter_bytes'] = (out/'synthetic_ONLY_scc_retrieval_lora.pth').stat().st_size
            report['queries'] = len(m['groups']); report['candidates'] = sum(len(g['candidates']) for g in m['groups'])
            report['passed'] = True
    except Exception as e:
        report['passed'] = False; report['error'] = repr(e)
        write_json(out/'lora_smoke_report.json', report)
        raise
    write_json(out/'lora_smoke_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cuda:0'); p.add_argument('--output', default='artifacts/lora_smoke')
    a = p.parse_args(); run(a.output, a.device)
