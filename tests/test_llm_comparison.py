import unittest
from src.agent.investigation_agent import InvestigationAgent
from src.evaluation.llm_comparison import compare_investigation_reports, numeric_correspondence
from src.llm.mock_provider import MockLLMProvider


class CapturingProvider:
    model = "offline-test-double"
    def generate(self, system, user, context):
        self.request = (system, user, context)
        return "O faturamento anterior foi 84.509,30. Confira a composição dos produtos."


class ComparisonTest(unittest.TestCase):
    def investigation(self):
        from src.data.adapters import load_sales_dataset
        data = load_sales_dataset("csv", csv_path="data/vendas_demo.csv")
        return InvestigationAgent(MockLLMProvider()).investigate("Compare 2024-07 e 2024-08.", data)

    def test_no_provider_means_not_run_and_pending_human_review(self):
        result = compare_investigation_reports(self.investigation())
        self.assertEqual(result["ai"]["status"], "not_run")
        self.assertEqual(result["human_review"]["status"], "pending")
        self.assertEqual(result["local"]["status"], "completed")

    def test_identical_context_prompts_and_error_redaction(self):
        provider = CapturingProvider()
        result = compare_investigation_reports(self.investigation(), provider)
        self.assertEqual(provider.request, (result["system_prompt"], result["user_prompt"], result["context"]))
        self.assertEqual(result["ai"]["status"], "completed")
        class FailingProvider:
            def generate(self, *args):
                raise RuntimeError("HTTP 403 Bearer gsk_TESTSECRET12345678901234567890")
        failed = compare_investigation_reports(self.investigation(), FailingProvider())
        self.assertEqual(failed["ai"]["status"], "failed")
        self.assertNotIn("gsk_TESTSECRET", failed["ai"]["error"])
        self.assertEqual(failed["local"]["status"], "completed")

    def test_numeric_check_accepts_br_us_and_ignores_dates_and_identifiers(self):
        text = "R$ 1.234,56 e USD 1,234.56; data 2024-08. ID abc1299083318993; total 999."
        check = numeric_correspondence(text, {"receita": 1234.56})
        self.assertEqual(check["total"], 3)
        self.assertEqual(check["matched"], 2)
        self.assertEqual(check["needs_review"], ["999"])

