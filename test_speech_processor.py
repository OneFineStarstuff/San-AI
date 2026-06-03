import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
from fastapi import UploadFile
from main import SpeechProcessor

class TestSpeechProcessor(unittest.TestCase):
    @patch('main.whisper.load_model')
    @patch('main.pyttsx3.init')
    def setUp(self, mock_tts_init, mock_whisper_load):
        self.speech_processor = SpeechProcessor()

    @patch('os.path.exists')
    @patch('os.remove')
    def test_speech_to_text(self, mock_remove, mock_exists):
        mock_exists.return_value = True
        # Mock the whisper model transcription
        self.speech_processor.whisper_model.transcribe = MagicMock(return_value={"text": "test transcription"})

        audio_content = BytesIO(b'Test audio content')
        audio_file = UploadFile(filename="test.wav", file=audio_content)

        result = self.speech_processor.speech_to_text(audio_file)
        self.assertEqual(result, "test transcription")

    def test_text_to_speech(self):
        text = "Hello world!"
        # Mock the say and runAndWait methods
        self.speech_processor.tts.say = MagicMock()
        self.speech_processor.tts.runAndWait = MagicMock()

        result = self.speech_processor.text_to_speech(text)
        self.assertIsNone(result)
        self.speech_processor.tts.say.assert_called_with(text)

    def test_text_to_speech_empty_text(self):
        with self.assertRaises(ValueError):
            self.speech_processor.text_to_speech("")

if __name__ == '__main__':
    unittest.main()
