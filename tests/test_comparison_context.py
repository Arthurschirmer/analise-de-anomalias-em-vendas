import unittest

import pandas as pd

from src.agent.comparison_context import comparison_context, resolve_comparison


class ComparisonContextTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"data": pd.to_datetime(["2023-12-10", "2024-01-10", "2024-02-10", "2024-03-10"])})
        self.context = {"previous": "2024-01", "current": "2024-02", "window_note": "Dias 1 a 9."}

    def test_followup_keeps_selected_comparison_not_latest_months(self):
        resolved = resolve_comparison("E por que isso ocorreu?", self.data, self.context)
        self.assertEqual(resolved["periods"], ["2024-01", "2024-02"])
        self.assertEqual(resolved["origin"], "context")

    def test_explicit_months_override_existing_comparison(self):
        self.assertEqual(resolve_comparison("Investigue fevereiro de 2024 e março de 2024", self.data, self.context)["periods"], ["2024-02", "2024-03"])

    def test_one_month_uses_observed_immediate_previous_month_and_handles_year_boundary(self):
        resolved = resolve_comparison("Por que janeiro de 2024 caiu?", self.data)
        self.assertEqual(resolved["periods"], ["2023-12", "2024-01"])
        self.assertEqual(resolved["origin"], "previous_month")

    def test_missing_context_does_not_guess_a_pair(self):
        self.assertIsNone(resolve_comparison("Por que isso ocorreu?", self.data)["periods"])

    def test_missing_months_and_more_than_two_months_require_clarification(self):
        missing = self.data.loc[self.data["data"].dt.month.ne(2)]
        self.assertEqual(resolve_comparison("Por que março de 2024 caiu?", missing)["reason"], "unavailable")
        self.assertEqual(resolve_comparison("Compare janeiro, fevereiro e março de 2024", self.data)["reason"], "too_many")
        self.assertIsNone(resolve_comparison("Por que isso ocorreu?", missing, self.context)["periods"])

    def test_window_note_is_carried_without_changing_periods(self):
        report = {"periodos": {"periodo_anterior": "2024-01", "periodo_atual": "2024-02"},
                  "contexto_da_base": {"comparison_window_definition": "Dias 1 a 9."}}
        self.assertEqual(comparison_context(report), self.context)


if __name__ == "__main__":
    unittest.main()
