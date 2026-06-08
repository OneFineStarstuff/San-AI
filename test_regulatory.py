import unittest
from main import RegulatoryModule, ZKFairnessProof, ContextualAttributionEnvelope

class TestRegulatoryModule(unittest.TestCase):
    def setUp(self):
        self.regulatory = RegulatoryModule()

    def test_verify_zk_fairness(self):
        result = self.regulatory.verify_zk_fairness("test input")
        self.assertIsInstance(result, ZKFairnessProof)
        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(result.demographic_parity_score, 0.98)

    def test_generate_cae(self):
        result = self.regulatory.generate_cae("TestModule", "test output")
        self.assertIsInstance(result, ContextualAttributionEnvelope)
        self.assertIn("TestModule", result.contribution_scores)
        self.assertEqual(result.contribution_scores["TestModule"], 1.0)

if __name__ == '__main__':
    unittest.main()
