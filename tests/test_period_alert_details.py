import unittest

import pandas as pd

from src.analysis.alert_explanation import explain_alert
from src.analysis.period_coverage import comparison_coverage, select_comparison_window, report_comparison_data
from src.ui.analytics import period_comparison


def calendar_sales():
    dates = list(pd.date_range("2011-10-01", "2011-10-31")) + list(pd.date_range("2011-11-01", "2011-11-09"))
    data = pd.DataFrame({"data": dates, "produto": "A", "categoria": "C", "regiao": "UK", "vendedor": "V",
                         "quantidade": 1, "preco_unitario": 10., "desconto": 0., "faturamento": 10.})
    data.attrs["dataset_context"] = {"currency": "GBP", "discount_available": False}
    return data


class PeriodAlertDetailsTest(unittest.TestCase):
    def test_same_day_window_changes_totals_without_changing_source(self):
        data = calendar_sales()
        full = period_comparison(data, "2011-10", "2011-11")
        self.assertAlmostEqual(full["variacao_percentual_faturamento"], -70.96774193548387)
        aligned = select_comparison_window(data, "2011-10", "2011-11", True)
        compared = period_comparison(aligned, "2011-10", "2011-11")
        self.assertEqual(compared["faturamento_anterior"], 90)
        self.assertEqual(compared["faturamento_atual"], 90)
        self.assertEqual(compared["variacao_faturamento"], 0)
        self.assertEqual(len(data), 40)
        self.assertNotIn("comparison_window", data.attrs["dataset_context"])
        window = aligned.attrs["dataset_context"]["comparison_window"]
        self.assertEqual((window["first_day"], window["last_day"]), (1, 9))

    def test_coverage_describes_observations_not_collection_certainty(self):
        data = calendar_sales()
        info = comparison_coverage(data, "2011-10", "2011-11")
        self.assertTrue(info["needs_attention"])
        self.assertEqual(info["current"]["observed_days"], 9)
        self.assertEqual(info["current"]["calendar_days"], 30)
        self.assertEqual(info["current"]["last_date"], "2011-11-09")

    def test_missing_days_are_not_filled_and_monthly_operations_are_not_mixed(self):
        data = calendar_sales()
        data = data.loc[data["data"].ne(pd.Timestamp("2011-11-05"))].copy()
        data.attrs["olist_raw_dir"] = "do-not-read"
        aligned = select_comparison_window(data, "2011-10", "2011-11", True)
        self.assertEqual(len(aligned), 17)
        self.assertNotIn("olist_raw_dir", aligned.attrs)
        report = {"contexto_da_base": aligned.attrs["dataset_context"]}
        pd.testing.assert_frame_equal(report_comparison_data(data, report), aligned, check_flags=False)

    def test_no_overlap_cannot_be_presented_as_a_comparison(self):
        data = calendar_sales()
        data = data.loc[(data["data"].dt.month == 11) | (data["data"].dt.day >= 20)]
        with self.assertRaises(ValueError):
            select_comparison_window(data, "2011-10", "2011-11", True)

    def test_alert_card_uses_calculated_reference_and_threshold(self):
        row = {"periodo": "2011-11-01", "valor": 600., "expected": 1000., "score": 8.,
               "n_history": 8, "eligible": True, "flag": True}
        facts = explain_alert(row, {"threshold": 3})
        self.assertEqual(facts["difference"], -400)
        self.assertEqual(facts["percentage"], -40)
        self.assertEqual(facts["direction"], "abaixo")
        self.assertEqual(facts["threshold"], 3)
        self.assertEqual(facts["previous_observations"], 8)

    def test_zero_or_negative_reference_does_not_get_an_invented_percentage(self):
        for expected in (0., -10., float("nan")):
            facts = explain_alert({"valor": 10., "expected": expected}, {"threshold": 3})
            self.assertIsNone(facts["percentage"])


if __name__ == "__main__":
    unittest.main()
