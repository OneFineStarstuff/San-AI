from datetime import datetime, timedelta, timezone
import io

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from main import ALGORITHM, SECRET_KEY, app, authenticate_user, create_access_token

client = TestClient(app)


@pytest.fixture
def auth_header():
    token = create_access_token({"sub": "testuser"})
    return {"Authorization": f"Bearer {token}"}


def test_nlp_endpoint(auth_header):
    response = client.post(
        "/process-nlp/",
        json={
            "text": "test"},
        headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "zk_proof" in data
    assert "cae_metadata" in data


def test_cv_endpoint(auth_header):
    image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = client.post(
        "/process-cv-detection/",
        files=files,
        headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "cae_metadata" in data


def test_stt_endpoint(auth_header):
    with open("test.wav", "rb") as f:
        files = {'file': ('test.wav', f, 'audio/wav')}
        response = client.post(
            "/speech-to-text/",
            files=files,
            headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "cae_metadata" in data

def test_nlp_endpoint_rejects_missing_bearer_token():
    response = client.post("/process-nlp/", json={"text": "test"})
    assert response.status_code == 401


def test_access_token_contains_subject_and_expiry():
    token = create_access_token({"sub": "testuser"})
    claims = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert claims["sub"] == "testuser"
    assert claims["exp"] > int(datetime.now(timezone.utc).timestamp())


def test_expired_access_token_is_rejected():
    token = jwt.encode(
        {
            "sub": "testuser",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=1),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    with pytest.raises(HTTPException) as exc_info:
        authenticate_user(token)
    assert exc_info.value.status_code == 401
