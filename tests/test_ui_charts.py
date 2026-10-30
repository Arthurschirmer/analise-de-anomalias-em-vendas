import json
import unittest

import pandas as pd

from src.ui.charts import change_chart, comparison_chart, contribution_chart, trend_chart


class ChartTest(unittest.TestCase):
    def setUp(self):
        self.monthly = pd.DataFrame([
            {"periodo": "2024-07", "periodo_label": "jul/2024", "valor": 10000,
             "anomalia": False, "variacao_percentual": None},
            {"periodo": "2024-08", "periodo_label": "ago/2024", "valor": 4000,
             "anomalia": True, "variacao_percentual": -60},
        ])

    def test_charts_validate_with_empty_data(self):
        charts = [trend_chart(pd.DataFrame()), change_chart(pd.DataFrame()),
                  contribution_chart(pd.DataFrame(), "julho", "agosto"), comparison_chart({})]
        for chart in charts:
            with self.subTest(chart=type(chart).__name__):
                spec = chart.to_dict(validate=True)
                self.assertEqual(spec["height"], 290)
                self.assertNotIn("background", spec)

    def test_one_period_keeps_visible_point_without_fake_monthly_change(self):
        trend_chart(self.monthly.head(1)).to_dict(validate=True)
        spec = change_chart(self.monthly.head(1)).to_dict(validate=True)
        datasets = list(spec.get("datasets", {}).values())
        self.assertTrue(any(dataset == [] for dataset in datasets))

    def test_currency_tooltips_and_source_are_preserved(self):
        original = self.monthly.copy(deep=True)
        spec = trend_chart(self.monthly).to_dict(validate=True)
        text = json.dumps(spec, ensure_ascii=False)
        self.assertIn("R$ 10.000,00", text)
        self.assertIn("Variação sinalizada", text)
        pd.testing.assert_frame_equal(self.monthly, original)

    def test_quantity_chart_uses_units_instead_of_currency(self):
        text = json.dumps(trend_chart(self.monthly, metric="quantidade").to_dict(), ensure_ascii=False)
        self.assertIn("Unidades vendidas", text)
        self.assertNotIn("R$", text)

    def test_missing_calendar_month_does_not_join_observations(self):
        monthly = pd.DataFrame([
            {"periodo": "2024-07", "valor": 10000, "variacao_percentual": None},
            {"periodo": "2024-09", "valor": 4000, "variacao_percentual": -60},
        ])
        spec = trend_chart(monthly).to_dict(validate=True)
        rows = next(values for values in spec["datasets"].values()
                    if values and "trecho" in values[0])
        by_month = {row["periodo"]: row for row in rows}
        self.assertIn("2024-08", by_month)
        self.assertIsNone(by_month["2024-08"]["valor"])
        self.assertNotEqual(by_month["2024-07"]["trecho"], by_month["2024-09"]["trecho"])
        self.assertEqual(spec["layer"][1]["encoding"]["detail"]["field"], "trecho")
        self.assertIn("ago/2024", spec["layer"][1]["encoding"]["x"]["scale"]["domain"])
        self.assertEqual(by_month["2024-09"]["sinal"], "Sem comparação mensal")

    def test_non_comparable_observation_is_not_shown_as_normal(self):
        monthly = self.monthly.copy()
        monthly["comparavel"] = [False, False]
        spec = trend_chart(monthly).to_dict(validate=True)
        rows = next(values for values in spec["datasets"].values() if values and "sinal" in values[0])
        self.assertTrue(all(row["sinal"] == "Sem comparação mensal" for row in rows))
        change_spec = change_chart(monthly).to_dict(validate=True)
        self.assertTrue(any(values == [] for values in change_spec["datasets"].values()))

    def test_contribution_and_comparison_validate_with_negative_changes(self):
        data = pd.DataFrame([{"segmento": "Notebook", "variacao_faturamento": -6000,
                              "faturamento_anterior": 10000, "faturamento_atual": 4000}])
        spec = contribution_chart(data, "jul/2024", "ago/2024").to_dict(validate=True)
        self.assertIn("R$ -6.000,00", json.dumps(spec, ensure_ascii=False))
        report = {"resumo_comparativo": {"faturamento_anterior": 10000, "faturamento_atual": 4000},
                  "periodos": {"periodo_anterior": "2024-07", "periodo_atual": "2024-08"}}
        spec = comparison_chart(report).to_dict(validate=True)
        self.assertIn("ago/2024", json.dumps(spec))


if __name__ == "__main__":
    unittest.main()
