"""
Unit tests for the CVModule.
"""

import unittest
from PIL import Image
from main import CVModule


class TestCVModule(unittest.TestCase):
    """Test suite for the CVModule."""

    def setUp(self):
        """Initializes the CV module for testing."""
        self.cv = CVModule()

    def test_detect_objects(self):
        """Tests object detection with a valid image."""
        # Create a dummy image for testing
        image = Image.new('RGB', (100, 100), color='white')
        result = self.cv.detect_objects(image)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_detect_objects_invalid_image(self):
        """Tests object detection with an invalid image."""
        with self.assertRaises(ValueError):
            self.cv.detect_objects(None)


if __name__ == '__main__':
    unittest.main()
