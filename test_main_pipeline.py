"""
Integration tests for the Enhanced AGI Pipeline.
"""

import io

import pytest
from fastapi import UploadFile
from PIL import Image

from main import EnhancedAGIPipeline


@pytest.mark.asyncio
async def test_pipeline_nlp():
    """Tests the pipeline's NLP processing capabilities."""
    pipeline = EnhancedAGIPipeline()
    result = await pipeline.process_nlp("Hello")
    assert "response" in result
    assert "zk_proof" in result
    assert "cae_metadata" in result
    assert result["zk_proof"].status == "VERIFIED"


@pytest.mark.asyncio
async def test_pipeline_cv():
    """Tests the pipeline's CV object detection capabilities."""
    pipeline = EnhancedAGIPipeline()
    image = Image.new("RGB", (100, 100), color="white")
    result = await pipeline.process_cv(image)
    assert "detections" in result
    assert "cae_metadata" in result


@pytest.mark.asyncio
async def test_pipeline_stt(audio_file):
    """Tests the pipeline's Speech-to-Text transcription capabilities."""
    pipeline = EnhancedAGIPipeline()

    with open(audio_file, "rb") as f:
        audio_file_obj = UploadFile(filename="test.wav", file=io.BytesIO(f.read()))
    result = await pipeline.process_speech_to_text(audio_file_obj)
    assert "response" in result
    assert "cae_metadata" in result
