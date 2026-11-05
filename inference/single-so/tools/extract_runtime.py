#!/usr/bin/env python3
"""Recover runtime.tar.gz from a previously assembled libvNavigatorInterface.so (see assemble.py).

Usage: python3 tools/extract_runtime.py /path/to/old/libvNavigatorInterface.so runtime.tar.gz
The footer is struct '<16sQQ64s': magic, payload offset, payload size, payload SHA256 (hex).
"""
import hashlib
import struct
import sys
from pathlib import Path

FOOTER = struct.Struct('<16sQQ64s')
so, out = map(Path, sys.argv[1:3])
size = so.stat().st_size
with so.open('rb') as f:
    f.seek(size - FOOTER.size)
    magic, offset, length, sha = FOOTER.unpack(f.read(FOOTER.size))
    if magic.rstrip(b'\0') != b'UAVLOC_SO_V1' or offset + length + FOOTER.size != size:
        raise SystemExit('not an assembled UAVLOC .so (no valid runtime footer)')
    f.seek(offset); digest = hashlib.sha256(); tmp = out.with_suffix('.partial')
    with tmp.open('wb') as dst:
        left = length
        while left:
            b = f.read(min(left, 4*1024*1024)); digest.update(b); dst.write(b); left -= len(b)
if digest.hexdigest() != sha.decode():
    tmp.unlink(); raise SystemExit('embedded runtime SHA256 mismatch')
tmp.replace(out)
print('runtime:', out, 'bytes:', length, 'sha256:', digest.hexdigest())
