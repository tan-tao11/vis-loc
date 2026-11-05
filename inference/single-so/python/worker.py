import argparse
import json
import math
import os
import socket
import struct
import sys
import traceback


def read_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        block = sock.recv(size-len(chunks))
        if not block:
            raise EOFError("adapter closed")
        chunks.extend(block)
    return chunks


def send_reply(sock, text):
    data = text.encode("utf-8", errors="replace")[:60000]
    sock.sendall(struct.pack("!I", len(data)) + data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fd", type=int)
    p.add_argument("--modality", type=int, choices=[0, 1])
    p.add_argument("--check-config", action="store_true")
    args = p.parse_args()
    from backends import load_config, Reference, AnyVisBackend, SCCBackend
    import numpy as np
    if args.check_config:
        try:
            config = load_config(args.config)
            for mode in config.get("enabled_modalities", [0, 1]):
                key = "rgb" if mode == 1 else "infrared"
                Reference(config[key]["reference"])
        except Exception as exc:
            print(f"CONFIG_ERROR: {exc}", file=sys.stderr)
            return 1
        print("CONFIG_OK")
        return
    if args.fd is None or args.modality is None:
        p.error("--fd and --modality required")
    with socket.socket(fileno=args.fd) as sock:
        try:
            config = load_config(args.config)
            if args.modality not in config.get("enabled_modalities", [0, 1]):
                send_reply(sock, "DISABLED")
                return
            if os.environ.get('NAV_EMBEDDED_ROOT'):
                from embedded_config import configure
                configure(config,args.modality)
            backend = (AnyVisBackend if args.modality == 1 else SCCBackend)(config)
            send_reply(sock, "READY")
        except Exception as exc:
            traceback.print_exc()
            send_reply(sock, f"ERROR initialization: {exc}")
            return
        while True:
            try:
                size, = struct.unpack("!I", read_exact(sock, 4))
                if not 0 < size <= 65536:
                    raise ValueError("invalid metadata length")
                request = json.loads(read_exact(sock, size))
                w, h, c = (int(request[k]) for k in ("width", "height", "channels"))
                if not (0 < w <= 32768 and 0 < h <= 32768 and c in (1, 3) and w*h*c <= 128*1024*1024):
                    raise ValueError("invalid image dimensions")
                pixels = read_exact(sock, w*h*c)
            except EOFError:
                return
            except Exception:
                traceback.print_exc()
                return  # Invalid framing cannot be safely resynchronized.
            try:
                backend.begin_frame(request)
                image = np.frombuffer(pixels, np.uint8).reshape((h, w) if c == 1 else (h, w, c))
                result = backend.localize(request, image)
                if len(result) != 4 or not all(math.isfinite(v) for v in result):
                    raise ValueError("nonfinite output")
                backend.end_frame(True)
                send_reply(sock, "OK " + " ".join(format(float(v), ".17g") for v in result))
            except Exception as exc:
                traceback.print_exc()
                backend.end_frame(False,str(exc))
                send_reply(sock, f"FAIL {exc}")


if __name__ == "__main__":
    sys.exit(main())
