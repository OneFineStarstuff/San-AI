"""
Shared fixtures for testing.
"""

import struct
import wave

import pytest


@pytest.fixture(name="audio_file")
def audio_file_fixture(tmp_path):
    """Fixture to create a dummy WAV file for testing."""
    file_path = tmp_path / "test.wav"
    # pylint: disable=no-member
    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(struct.pack("<h", 0) * 16000)
    return str(file_path)
