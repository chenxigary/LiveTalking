import os
import unittest
from unittest.mock import patch

from utils.device import initialize_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cpu_override_does_not_depend_on_accelerator_availability(self):
        with patch.dict(os.environ, {"V3_AVATAR_DEVICE": "cpu"}, clear=False):
            self.assertEqual(initialize_device().type, "cpu")

    def test_invalid_override_fails_fast(self):
        with patch.dict(os.environ, {"V3_AVATAR_DEVICE": "quantum"}, clear=False):
            with self.assertRaisesRegex(ValueError, "V3_AVATAR_DEVICE"):
                initialize_device()


if __name__ == "__main__":
    unittest.main()
