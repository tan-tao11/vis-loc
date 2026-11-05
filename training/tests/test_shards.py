import numpy as np
import pytest
import extract_features as extraction
from cache import group_cache_name
from common import write_json, read_json

def test_shard_merge_complete_missing_duplicate_stale(tmp_path, monkeypatch):
    groups = [{'id': f'DS/sequence{i}/1', 'candidates': [{}, {}]} for i in range(4)]
    manifest = tmp_path/'manifest.json'; write_json(manifest, {'manifest_id': 'M', 'groups': groups})
    monkeypatch.setattr(extraction, 'verify_manifest', lambda m: None)
    monkeypatch.setattr(extraction, 'backbone_identity', lambda c: {'name': 'fixture'})
    monkeypatch.setattr(extraction, 'extraction_identity', lambda *args: 'fingerprint')
    for i in range(2):
        write_json(tmp_path/f'shard_{i}.json', {'num_shards': 2, 'fingerprint': 'fingerprint', 'failures': [],
            'assigned_ids': [g['id'] for j, g in enumerate(groups) if j % 2 == i]})
    for g in groups:
        np.savez(tmp_path/group_cache_name(g['id']), id=g['id'], fingerprint='fingerprint',
                 query=np.ones(1024, np.float32), satellite=np.ones((2, 1024), np.float32), seconds=.1)
    extraction.merge({}, manifest, tmp_path, 2)
    assert read_json(tmp_path/'index.json')['group_ids'] == [g['id'] for g in groups]
    bad = read_json(tmp_path/'shard_1.json'); bad['assigned_ids'][0] = groups[0]['id']; write_json(tmp_path/'shard_1.json', bad)
    with pytest.raises(ValueError, match='partition'): extraction.merge({}, manifest, tmp_path, 2)
    bad['assigned_ids'][0] = groups[1]['id']; bad['fingerprint'] = 'old'; write_json(tmp_path/'shard_1.json', bad)
    with pytest.raises(ValueError, match='Stale'): extraction.merge({}, manifest, tmp_path, 2)
    bad['fingerprint'] = 'fingerprint'; write_json(tmp_path/'shard_1.json', bad)
    (tmp_path/group_cache_name(groups[0]['id'])).unlink()
    with pytest.raises(FileNotFoundError): extraction.merge({}, manifest, tmp_path, 2)
