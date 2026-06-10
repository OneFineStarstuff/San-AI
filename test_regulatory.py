import unittest
from main import RegulatoryModule, ZKFairnessProof, ContextualAttributionEnvelope

class TestRegulatoryModule(unittest.TestCase):
    def setUp(self):
        self.regulatory = RegulatoryModule()

    def test_verify_zk_fairness(self):
        input_text = "test input"
        result = self.regulatory.verify_zk_fairness(input_text)
        self.assertIsInstance(result, ZKFairnessProof)
        self.assertIn(result.status, ["VERIFIED", "FAILED"])
        # Based on new logic: 0.95 + (10 % 5) / 100 = 0.95
        self.assertEqual(result.demographic_parity_score, 0.95)
        self.assertTrue(result.proof_hash.startswith("zkp_"))

    def test_generate_cae(self):
        result = self.regulatory.generate_cae("TestModule", "test output")
        self.assertIsInstance(result, ContextualAttributionEnvelope)
        self.assertIn("TestModule", result.contribution_scores)
        self.assertEqual(result.contribution_scores["TestModule"], 0.85)
        # Check for presence of interpretability summary content
        self.assertIn("processed by TestModule", result.interpretability_summary)

if __name__ == '__main__':
    unittest.main()
