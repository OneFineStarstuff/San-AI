"""
Unit tests for the SpeechProcessor module.
"""

import unittest
from io import BytesIO
from fastapi import UploadFile
from main import SpeechProcessor


class TestSpeechProcessor(unittest.TestCase):
    """Test suite for the SpeechProcessor."""

    def setUp(self):
        """Initializes the speech processor for testing."""
        self.speech_processor = SpeechProcessor()

    def test_speech_to_text(self):
        """Tests transcription of speech to text."""
        # Create a dummy audio file for testing
        audio_content = BytesIO(b'Test audio content')
        audio_file = UploadFile(filename="test.wav", file=audio_content)
        result = self.speech_processor.speech_to_text(audio_file)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_text_to_speech(self):
        """Tests text-to-speech synthesis."""
        text = "Hello world!"
        # pylint: disable=assignment-from-no-return
        result = self.speech_processor.text_to_speech(text)
        self.assertIsNone(result)  # Text-to-speech returns None

    def test_text_to_speech_empty_text(self):
        """
        Tests text-to-speech with an empty string, expecting a ValueError.
        """
        with self.assertRaises(ValueError):
            self.speech_processor.text_to_speech("")


if __name__ == '__main__':
    unittest.main()
