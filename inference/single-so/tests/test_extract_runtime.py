import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
TOOLS = Path(__file__).resolve().parents[1]/'tools'


class ExtractRuntimeTests(unittest.TestCase):
    def test_round_trip_and_rejects_plain_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); (d/'base.so').write_bytes(os.urandom(4096)); (d/'rt.tar.gz').write_bytes(os.urandom(100000))
            subprocess.run([sys.executable, TOOLS/'assemble.py', d/'base.so', d/'rt.tar.gz', d/'out.so'], check=True, capture_output=True)
            subprocess.run([sys.executable, TOOLS/'extract_runtime.py', d/'out.so', d/'back.tar.gz'], check=True, capture_output=True)
            self.assertEqual((d/'back.tar.gz').read_bytes(), (d/'rt.tar.gz').read_bytes())
            r = subprocess.run([sys.executable, TOOLS/'extract_runtime.py', d/'base.so', d/'x.tar.gz'], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0); self.assertIn('footer', r.stderr)


if __name__ == '__main__':
    unittest.main()
