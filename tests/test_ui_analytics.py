import unittest

import pandas as pd

from src.ui.analytics import (
    available_periods,
    dimension_contributions,
    monthly_overview,
    period_comparison,
    period_label,
)


def sales(rows):
    frame = pd.DataFrame(rows, columns=["data", "produto", "faturamento"])
    frame["data"] = pd.to_datetime(frame["data"])
    frame["quantidade"] = 1
    frame["preco_unitario"] = frame["faturamento"]
    frame["desconto"] = 0.0
    frame["categoria"] = "Exemplo"
    frame["regiao"] = "Sul"
    frame["vendedor"] = "Ana"
    return frame


class DashboardAnalyticsTest(unittest.TestCase):
    def test_demo_monthly_totals_and_actual_percentage_flag(self):
        demo = pd.read_csv("data/vendas_demo.csv", parse_dates=["data"])
        frame = monthly_overview(demo)
        self.assertEqual(frame["periodo"].tolist(), ["2024-07", "2024-08"])
        self.assertAlmostEqual(frame.iloc[0]["valor"], 84509.3)
        self.assertAlmostEqual(frame.iloc[1]["valor"], 41411.7)
        self.assertAlmostEqual(frame.iloc[1]["variacao_percentual"], -50.9974641844)
        self.assertFalse(frame.iloc[0]["comparavel"])
        self.assertFalse(frame.iloc[0]["anomalia"])
        self.assertTrue(frame.iloc[1]["anomalia"])
        self.assertFalse(monthly_overview(demo, threshold_pct=60).iloc[1]["anomalia"])

    def test_zero_baseline_does_not_become_zero_percent_change(self):
        data = sales([("2024-07-01", "A", 0), ("2024-08-01", "A", 100)])
        row = monthly_overview(data).iloc[1]
        self.assertTrue(row["comparavel"])
        self.assertEqual(row["variacao"], 100)
        self.assertTrue(pd.isna(row["variacao_percentual"]))
        self.assertFalse(row["anomalia"])
        comparison = period_comparison(data)
        self.assertTrue(comparison["available"])
        self.assertIsNone(comparison["variacao_percentual_faturamento"])

    def test_missing_calendar_baseline_is_unavailable(self):
        data = sales([("2024-07-01", "A", 100), ("2024-09-01", "A", 20)])
        row = monthly_overview(data).iloc[1]
        self.assertFalse(row["comparavel"])
        self.assertFalse(row["anomalia"])
        result = period_comparison(data)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "missing_previous_period")
        self.assertIsNone(result["variacao_faturamento"])
        self.assertTrue(period_comparison(data, "2024-07", "2024-09")["available"])

    def test_empty_single_period_and_invalid_order_are_explicit(self):
        self.assertEqual(available_periods(pd.DataFrame()), [])
        self.assertTrue(monthly_overview(pd.DataFrame()).empty)
        self.assertEqual(period_comparison(pd.DataFrame())["reason"], "no_data")
        data = sales([("2024-07-01", "A", 100)])
        self.assertFalse(period_comparison(data)["available"])
        self.assertEqual(period_comparison(data, "2024-07", "2024-07")["reason"], "invalid_period_order")
        self.assertEqual(period_label("2024-07"), "jul/2024")

    def test_signed_contributions_use_whole_change_and_preserve_offsets(self):
        data = sales([
            ("2024-07-01", "A", 100), ("2024-07-02", "B", 100),
            ("2024-08-01", "A", 40), ("2024-08-02", "B", 140),
        ])
        frame = dimension_contributions(data, "2024-07", "2024-08", limit=None)
        self.assertEqual(frame["segmento"].tolist(), ["A", "B"])
        self.assertEqual(frame["variacao_faturamento"].tolist(), [-60, 40])
        self.assertEqual(frame["participacao_variacao_percentual"].tolist(), [300, -200])
        self.assertEqual(frame["variacao_faturamento"].sum(), -20)
        self.assertEqual(frame["participacao_variacao_percentual"].sum(), 100)
        top = dimension_contributions(data, "2024-07", "2024-08", limit=1)
        self.assertEqual(top.iloc[0]["participacao_variacao_percentual"], 300)

    def test_zero_total_and_disappearing_segment_avoid_false_average_changes(self):
        data = sales([
            ("2024-07-01", "A", 100), ("2024-08-01", "B", 100),
        ])
        frame = dimension_contributions(data, "2024-07", "2024-08")
        self.assertTrue(frame["participacao_variacao_percentual"].isna().all())
        self.assertTrue(frame["variacao_preco_unitario_medio"].isna().all())
        self.assertEqual(set(frame["direcao"]), {"queda", "aumento"})
        self.assertTrue(dimension_contributions(data, "2024-06", "2024-08").empty)
        with self.assertRaises(ValueError):
            dimension_contributions(data, "2024-07", "2024-08", dimension="lucro")


if __name__ == "__main__":
    unittest.main()
