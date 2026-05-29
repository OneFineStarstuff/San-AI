"""
Integration tests for the FastAPI application endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from main import app, EnhancedAGIPipeline


class TestMain(unittest.TestCase):
    """Test suite for the main FastAPI application."""

    def setUp(self):
        """Sets up the test client and pipeline instance."""
        self.client = TestClient(app)
        self.pipeline = EnhancedAGIPipeline()

    def test_process_nlp(self):
        """
        Tests the NLP processing endpoint.
        Note: Requires auth bypass or valid token in real scenarios.
        """
        # In a real scenario, we would mock authenticate_user
        # or provide a valid token.
        response = self.client.post("/process-nlp/", json={"text": "Hello"})
        # Expect 401 if token is missing
        self.assertEqual(response.status_code, 401)

    def test_text_to_speech(self):
        """Tests the text-to-speech endpoint. Expects 401 without auth."""
        response = self.client.post("/text-to-speech/", json={"text": "Hi"})
        self.assertEqual(response.status_code, 401)


if __name__ == '__main__':
    unittest.main()
