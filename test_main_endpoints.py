"""
Unit tests for FastAPI endpoints.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app, create_access_token

client = TestClient(app)


@pytest.fixture(name="auth_header")
def auth_header_fixture():
    """Fixture for generating authentication headers."""
    token = create_access_token({"sub": "testuser"})
    return {"Authorization": f"Bearer {token}"}


def test_nlp_endpoint(auth_header):
    """Tests the NLP processing endpoint."""
    response = client.post("/process-nlp/", json={"text": "test"}, headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "zk_proof" in data
    assert "cae_metadata" in data


def test_cv_endpoint(auth_header):
    """Tests the Computer Vision object detection endpoint."""
    image = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    files = {"file": ("test.png", img_byte_arr, "image/png")}
    response = client.post("/process-cv-detection/", files=files, headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "cae_metadata" in data


def test_stt_endpoint(auth_header, audio_file):
    """Tests the Speech-to-Text transcription endpoint."""
    with open(audio_file, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = client.post("/speech-to-text/", files=files, headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "cae_metadata" in data
