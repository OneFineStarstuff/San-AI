import unittest
from PIL import Image
from unittest.mock import MagicMock, patch
from main import CVModule

class TestCVModule(unittest.TestCase):
    @patch('main.YOLO')
    def setUp(self, mock_yolo):
        self.cv = CVModule()

    def test_detect_objects(self):
        # Create a dummy image for testing
        image = Image.new('RGB', (100, 100), color = 'white')

        # Mock YOLO model result
        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"detections": []}'
        self.cv.model.return_value = [mock_result]

        result = self.cv.detect_objects(image)
        self.assertIsNotNone(result)
        self.assertEqual(result, '{"detections": []}')

    def test_detect_objects_invalid_image(self):
        with self.assertRaises(ValueError):
            self.cv.detect_objects(None)

if __name__ == '__main__':
    unittest.main()
