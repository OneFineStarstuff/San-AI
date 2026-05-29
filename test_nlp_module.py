"""
Unit tests for the NLPModule.
"""

import unittest
from main import NLPModule


class TestNLPModule(unittest.TestCase):
    """Test suite for the NLPModule."""

    def setUp(self):
        """Initializes the NLP module for testing."""
        self.nlp = NLPModule()

    def test_generate_text(self):
        """Tests text generation with a valid prompt."""
        result = self.nlp.generate_text("Hello")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
