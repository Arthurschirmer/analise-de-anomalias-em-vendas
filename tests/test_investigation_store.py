import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.agent.investigation_agent import InvestigationAgent
from src.evaluation.investigation_log import InvestigationLog
from src.evaluation.investigation_store import dataset_fingerprint, save_investigation, load_investigation, list_investigations
from src.data.loader import load_sales_csv
from src.llm.mock_provider import MockLLMProvider


class InvestigationStoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.data = load_sales_csv("data/vendas_demo.csv")
        self.log = InvestigationAgent(MockLLMProvider()).investigate("Compare 2024-07 e 2024-08.", self.data)

    def test_snapshot_roundtrip_does_not_execute_a_tool_or_provider(self):
        entry = save_investigation(self.log, self.data, self.folder.name, label="Demonstração")
        with patch("src.tools.sales_tools.SalesAnalysisTools.run", side_effect=AssertionError("Must not execute")), patch.object(MockLLMProvider, "generate", side_effect=AssertionError("Must not generate")):
            stored = load_investigation(self.folder.name, entry["id"], dataset_fingerprint(self.data))
            restored = InvestigationLog.from_dict(stored["record"])
        self.assertEqual(restored.structured_report, self.log.structured_report)
        self.assertEqual(restored.final_conclusion, self.log.final_conclusion)
        self.assertEqual(restored.created_at, self.log.created_at)
        self.assertEqual(len(restored.tool_results), len(self.log.tool_results))

    def test_numeric_or_currency_change_prevents_reopening_as_the_same_dataset(self):
        entry = save_investigation(self.log, self.data, self.folder.name)
        changed = self.data.copy()
        changed.loc[0, "faturamento"] += 1
        self.assertNotEqual(dataset_fingerprint(changed), dataset_fingerprint(self.data))
        changed.attrs["dataset_context"] = {"currency": "GBP"}
        with self.assertRaisesRegex(ValueError, "outra base"):
            load_investigation(self.folder.name, entry["id"], dataset_fingerprint(changed))

    def test_invalid_records_do_not_hide_valid_history_and_no_temporary_file_remains(self):
        entry = save_investigation(self.log, self.data, self.folder.name)
        invalid_path = Path(self.folder.name, "f" * 32 + ".json")
        invalid_path.write_text("invalid json", encoding="utf-8")
        entries, failures = list_investigations(self.folder.name)
        self.assertEqual([item["id"] for item in entries], [entry["id"]])
        self.assertEqual(failures, 1)
        self.assertFalse(list(Path(self.folder.name).glob("*.tmp")))
        with self.assertRaises(ValueError):
            load_investigation(self.folder.name, "../outside")

    def test_nonfinite_optional_results_are_saved_as_json_null(self):
        self.log.evidence["optional"] = float("nan")
        entry = save_investigation(self.log, self.data, self.folder.name)
        self.assertIsNone(entry["record"]["evidence"]["optional"])
        payload = Path(self.folder.name, entry["id"] + ".json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", payload)
        self.assertIsNone(json.loads(payload)["record"]["evidence"]["optional"])


if __name__ == "__main__":
    unittest.main()
