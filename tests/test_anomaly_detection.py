import unittest

import pandas as pd

from src.analysis.anomaly_detection import (
    build_anomaly_description,
    detect_period_anomalies,
    get_most_relevant_anomaly,
)


class AnomalyDetectionTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [
                {
                    "data": "2024-07-01",
                    "produto": "Notebook",
                    "categoria": "Eletronicos",
                    "regiao": "Sul",
                    "vendedor": "Ana",
                    "quantidade": 10,
                    "preco_unitario": 3000,
                    "desconto": 0.0,
                    "faturamento": 30000,
                },
                {
                    "data": "2024-08-01",
                    "produto": "Notebook",
                    "categoria": "Eletronicos",
                    "regiao": "Sul",
                    "vendedor": "Ana",
                    "quantidade": 4,
                    "preco_unitario": 3000,
                    "desconto": 0.0,
                    "faturamento": 12000,
                },
            ]
        )
        self.data["data"] = pd.to_datetime(self.data["data"])

    def test_detect_period_anomalies(self):
        anomalies = detect_period_anomalies(self.data, threshold_pct=30)

        self.assertEqual(len(anomalies), 1)
        self.assertTrue(anomalies[0]["anomalia"])
        self.assertEqual(anomalies[0]["direcao"], "queda")
        self.assertEqual(anomalies[0]["metodos_disparados"], ["percentual"])

    def test_get_most_relevant_anomaly(self):
        anomalies = detect_period_anomalies(self.data, threshold_pct=30)
        anomaly = get_most_relevant_anomaly(anomalies)

        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly["periodo_atual"], "2024-08")

    def test_build_anomaly_description(self):
        anomalies = detect_period_anomalies(self.data, threshold_pct=30)
        anomaly = get_most_relevant_anomaly(anomalies)
        description = build_anomaly_description(anomaly)

        self.assertIn("agosto", description)
        self.assertIn("julho", description)
        self.assertIn("2024", description)

    def test_detect_anomaly_with_zscore(self):
        data = _build_monthly_data([1000, 1020, 980, 1010, 600])

        anomalies = detect_period_anomalies(
            data,
            method="zscore",
            zscore_threshold=2,
        )

        self.assertTrue(anomalies[-1]["anomalia"])
        self.assertIn("zscore", anomalies[-1]["metodos_disparados"])

    def test_detect_anomaly_with_iqr(self):
        data = _build_monthly_data([1000, 1010, 990, 1005, 500])

        anomalies = detect_period_anomalies(data, method="iqr")

        self.assertTrue(anomalies[-1]["anomalia"])
        self.assertIn("iqr", anomalies[-1]["metodos_disparados"])

    def test_detect_anomaly_with_moving_average(self):
        data = _build_monthly_data([1000, 1000, 1000, 400])

        anomalies = detect_period_anomalies(
            data,
            method="media_movel",
            moving_average_window=3,
            threshold_pct=30,
        )

        self.assertTrue(anomalies[-1]["anomalia"])
        self.assertIn("media_movel", anomalies[-1]["metodos_disparados"])

    def test_detect_anomaly_with_combined_method(self):
        data = _build_monthly_data([1000, 1000, 1000, 400])

        anomalies = detect_period_anomalies(
            data,
            method="combinado",
            moving_average_window=3,
            threshold_pct=30,
        )

        self.assertTrue(anomalies[-1]["anomalia"])
        self.assertIn("percentual", anomalies[-1]["metodos_disparados"])
        self.assertIn("media_movel", anomalies[-1]["metodos_disparados"])

    def test_min_period_value_avoids_tiny_baseline_anomaly(self):
        data = _build_monthly_data([10, 10000])

        anomalies = detect_period_anomalies(
            data,
            threshold_pct=30,
            min_period_value=1000,
        )

        self.assertFalse(anomalies[0]["anomalia"])
        self.assertFalse(anomalies[0]["base_suficiente"])

    def test_missing_calendar_month_is_not_compared_as_adjacent(self):
        data = _build_monthly_data([1000, 100, 300])
        data["data"] = pd.to_datetime(["2024-01-01", "2024-03-01", "2024-04-01"])
        anomalies = detect_period_anomalies(data, threshold_pct=30)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["periodo_anterior"], "2024-03")
        self.assertEqual(anomalies[0]["periodo_atual"], "2024-04")
        self.assertEqual(anomalies[0]["variacao_percentual"], 200)

    def test_statistical_history_restarts_after_missing_month(self):
        data = _build_monthly_data([1000, 1000, 1000, 100])
        data["data"] = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01"])
        anomalies = detect_period_anomalies(data, method="zscore")
        self.assertEqual(len(anomalies), 2)
        self.assertIsNone(anomalies[-1]["z_score"])
        self.assertFalse(anomalies[-1]["anomalia"])


def _build_monthly_data(values):
    rows = []
    for index, revenue in enumerate(values, start=1):
        rows.append(
            {
                "data": f"2024-{index:02d}-01",
                "produto": "Produto Teste",
                "categoria": "Categoria Teste",
                "regiao": "Sul",
                "vendedor": "Ana",
                "quantidade": 10,
                "preco_unitario": revenue / 10,
                "desconto": 0.0,
                "faturamento": revenue,
            }
        )

    data = pd.DataFrame(rows)
    data["data"] = pd.to_datetime(data["data"])
    return data


if __name__ == "__main__":
    unittest.main()
