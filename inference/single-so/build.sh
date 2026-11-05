#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $(uname -m) != aarch64 ]]; then echo 'Build on ARM64 Ubuntu matching the competition machine.' >&2; exit 1; fi
if [[ ! -f runtime.tar.gz ]]; then echo 'Missing runtime.tar.gz; run tools/capture_runtime.py on the prepared source board first.' >&2; exit 1; fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python3 tools/assemble.py build/lib/libvNavigatorInterface.so runtime.tar.gz dist/libvNavigatorInterface.so
sha256sum dist/libvNavigatorInterface.so > dist/SHA256SUMS
echo 'Do not strip/objcopy the final .so: the appended runtime must be retained.'
