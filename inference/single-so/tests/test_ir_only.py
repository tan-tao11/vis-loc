import socket
import struct
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))


class InfraredOnlyTests(unittest.TestCase):
    def test_rgb_worker_reports_disabled_before_reading_config(self):
        a, b = socket.socketpair()
        p = subprocess.Popen([sys.executable, str(ROOT/'python/worker.py'), '--fd', str(b.fileno()), '--modality', '1',
                              '--config', '/nonexistent/naviParams.txt'], pass_fds=[b.fileno()])
        b.close()
        try:
            n = struct.unpack('!I', a.recv(4))[0]
            self.assertEqual(a.recv(n).decode(), 'DISABLED')
        finally:
            a.close(); p.wait(timeout=30)

    def test_config_must_be_infrared_only(self):
        import tempfile, yaml
        from backends import load_config
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'naviParams.txt'
            p.write_text(yaml.safe_dump(dict(version=1, enabled_modalities=[0, 1], infrared={})))
            with self.assertRaisesRegex(ValueError, 'only infrared'):
                load_config(p)


if __name__ == '__main__':
    unittest.main()


class SharedSoConfigTests(unittest.TestCase):
    def test_vertical_and_slant_examples_share_one_build(self):
        from tuning import read_config
        import consistency, self_eval
        from prior_tracker import PriorTracker
        v = read_config(ROOT.parent/'naviParams.vertical.example.txt')['infrared']
        s = read_config(ROOT.parent/'naviParams.slant.example.txt')['infrared']
        self.assertTrue(v['camera']['nadir_reference_alignment']); self.assertFalse(v['camera'].get('oblique', False))
        self.assertFalse(s['camera']['nadir_reference_alignment']); self.assertTrue(s['camera']['oblique'])
        self.assertNotEqual(v['retrieval_head'], s['retrieval_head'])
        for sec in (v, s):   # every section parses with this code
            consistency.settings(sec['consistency']); self_eval.settings(sec['self_eval']); PriorTracker(sec['prior_tracking'])
