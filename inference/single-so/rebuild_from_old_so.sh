#!/usr/bin/env bash
# Rebuild libvNavigatorInterface.so with the current python/ and projects/ sources, reusing the
# embedded runtime (Python env, wheels, native libs) of a previously working .so.
#   bash rebuild_from_old_so.sh /path/to/old/libvNavigatorInterface.so
# Output: dist/libvNavigatorInterface.so (+ dist/SHA256SUMS). Run on ARM64 Ubuntu (same as the competition machine).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
OLD=${1:?usage: bash rebuild_from_old_so.sh /path/to/old/libvNavigatorInterface.so}
[[ -f $OLD ]] || { echo "missing $OLD" >&2; exit 1; }
[[ $(uname -m) == aarch64 ]] || { echo 'Build on ARM64 Ubuntu matching the competition machine.' >&2; exit 1; }
python3 tools/extract_runtime.py "$OLD" runtime.tar.gz      # 1. recover the embedded runtime (SHA256 checked)
rm -rf runtime-stage                                          # 2. never reuse a stale unpacked runtime
python3 tools/repack_runtime.py                               # 3. put the new adapter/ and projects/ into it
bash build.sh                                                 # 4. compile + append runtime -> dist/
cmake --build build -j2 --target runtime_check >/dev/null 2>&1 || true
if [[ -x build/runtime_check ]]; then build/runtime_check dist/libvNavigatorInterface.so && echo RUNTIME_OK; fi
echo "Built dist/libvNavigatorInterface.so; keep $OLD as a fallback."
