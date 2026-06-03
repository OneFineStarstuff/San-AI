import unittest
from unittest.mock import MagicMock, patch
from main import NLPModule

class TestNLPModule(unittest.TestCase):
    @patch('main.T5Tokenizer.from_pretrained')
    @patch('main.T5ForConditionalGeneration.from_pretrained')
    def setUp(self, mock_model, mock_tokenizer):
        self.nlp = NLPModule()

    def test_generate_text(self):
        # Mock tokenizer and model behavior
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        self.nlp.tokenizer.return_value = mock_inputs
        self.nlp.model.generate.return_value = [MagicMock()]
        self.nlp.tokenizer.decode.return_value = "Generated text"

        result = self.nlp.generate_text("Hello")
        self.assertEqual(result, "Generated text")

    def test_generate_text_empty_prompt(self):
        with self.assertRaises(ValueError):
            self.nlp.generate_text("")

if __name__ == '__main__':
    unittest.main()
