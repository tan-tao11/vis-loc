#!/usr/bin/env python3
"""Read-only comparison of this source snapshot and an extracted runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
    result = {'matched': 0, 'missing': [], 'different': [], 'repository_modified': []}
    for item in manifest['files']:
        if 'runtime_path' not in item:
            continue
        src, dest = root/item['path'], args.cache/item['runtime_path']
        if digest(src) != item['sha256']:
            result['repository_modified'].append(item['path'])
        if not dest.is_file():
            result['missing'].append(item['runtime_path'])
        elif digest(dest) != item['sha256']:
            result['different'].append(item['runtime_path'])
        else:
            result['matched'] += 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(result[k] for k in ('missing', 'different', 'repository_modified')))


if __name__ == '__main__':
    raise SystemExit(main())
