import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.device import select_device, whisper_options


class DeviceTests(unittest.TestCase):
    def test_auto_priority_and_cpu_fallback(self):
        for cuda, mps, expected in [(True, True, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
            with self.subTest(expected=expected), patch('torch.cuda.is_available', return_value=cuda), patch('torch.backends.mps.is_available', return_value=mps):
                self.assertEqual(select_device('auto'), expected)
                self.assertEqual(select_device('cpu'), 'cpu')

    def test_explicit_unavailable_device_fails(self):
        with patch('torch.cuda.is_available', return_value=False), patch('torch.backends.mps.is_available', return_value=False):
            for device in ('cuda', 'mps', 'invalid'):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    select_device(device)

    def test_whisper_never_receives_mps(self):
        self.assertEqual(whisper_options('mps'), {'device': 'cpu', 'compute_type': 'int8'})
        self.assertEqual(whisper_options('cpu'), {'device': 'cpu', 'compute_type': 'int8'})
        self.assertEqual(whisper_options('cuda'), {'device': 'cuda', 'compute_type': 'float16'})


if __name__ == '__main__':
    unittest.main()
