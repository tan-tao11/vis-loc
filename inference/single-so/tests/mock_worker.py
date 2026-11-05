"""Transport-only fixture. Never used by production configuration."""
import argparse
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"python"))
from worker import read_exact, send_reply
p=argparse.ArgumentParser()
p.add_argument('--fd', type=int)
p.add_argument('--modality', type=int)
p.add_argument('--config')
a=p.parse_args()
with socket.socket(fileno=a.fd) as s:
    if os.getenv('TEST_BEHAVIOR')=='init_fail':
        send_reply(s,'ERROR missing map fixture'); sys.exit()
    send_reply(s,'READY')
    while True:
        try:
            size,=struct.unpack('!I',read_exact(s,4))
            r=json.loads(read_exact(s,size))
            image=read_exact(s,r['width']*r['height']*r['channels'])
        except EOFError: break
        assert r['imgType']==a.modality
        assert len(r['telemetry'])==15 and not any('_gt' in k for k in r)
        assert len(image)>0
        if os.getenv('TEST_BEHAVIOR')=='crash': sys.exit(9)
        if os.getenv('TEST_BEHAVIOR')=='timeout': time.sleep(5)
        if os.getenv('TEST_BEHAVIOR')=='bad_result': send_reply(s,'OK nan 30 100 1'); continue
        if os.getenv('TEST_BEHAVIOR')=='fail': send_reply(s,'FAIL no PnP'); continue
        send_reply(s,f'OK {110+a.modality} 30 {100+r["number"]} 0')
