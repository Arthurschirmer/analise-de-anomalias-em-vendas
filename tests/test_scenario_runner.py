import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.scenario_runner import load_scenarios, run_scenarios
from src.llm.mock_provider import MockLLMProvider


class ScenarioRunnerTest(unittest.TestCase):
    def test_load_scenarios(self):
        scenarios = load_scenarios("data/scenarios.json")
        self.assertGreaterEqual(len(scenarios), 3)
        self.assertIn("anomalia", scenarios[0])

    @patch("src.evaluation.scenario_runner.create_llm_provider_from_env", return_value=MockLLMProvider())
    def test_run_scenarios_creates_summary(self, provider_factory):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path, summary = run_scenarios("data/scenarios.json", temp_dir)

            self.assertTrue(Path(summary_path).exists())
            self.assertEqual(len(summary), 4)
            self.assertIn("acerto_top1", summary[0])
            self.assertEqual(summary[0]["acerto_top1"], "sim")
            self.assertIn("tempo_execucao_segundos", summary[0])

            first_log = Path(summary[0]["log"])
            self.assertTrue(first_log.exists())
            with open(first_log, "r", encoding="utf-8") as file:
                log_data = json.load(file)
            self.assertIn("final_conclusion", log_data)
            self.assertIn("evaluation", log_data)

            evaluation_summary = Path(temp_dir) / "evaluation_summary.json"
            evaluation_report = Path(temp_dir) / "evaluation_report.md"
            self.assertTrue(evaluation_summary.exists())
            self.assertTrue(evaluation_report.exists())


if __name__ == "__main__":
    unittest.main()
