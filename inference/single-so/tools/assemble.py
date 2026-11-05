#!/usr/bin/env python3
"""Append the runtime after the ELF image, outside LOAD segments."""
import hashlib
from pathlib import Path
import shutil
import struct
import sys

base,payload,out=map(Path,sys.argv[1:4])
out.parent.mkdir(parents=True,exist_ok=True)
digest=hashlib.sha256()
tmp=out.with_suffix('.so.partial')
with base.open('rb') as src,tmp.open('wb') as dst:
    shutil.copyfileobj(src,dst,4*1024*1024)
    offset=dst.tell()
    with payload.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            digest.update(b);dst.write(b)
    dst.write(struct.pack('<16sQQ64s',b'UAVLOC_SO_V1'.ljust(16,b'\0'),offset,payload.stat().st_size,digest.hexdigest().encode()))
tmp.chmod(0o755);tmp.replace(out)
print('SO:',out,'bytes:',out.stat().st_size,'payload_sha256:',digest.hexdigest())
