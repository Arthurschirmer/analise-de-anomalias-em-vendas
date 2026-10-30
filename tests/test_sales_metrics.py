import unittest

import pandas as pd

from src.analysis.sales_metrics import (
    compare_periods,
    compare_periods_by_dimension,
    percentage_change,
    total_quantity,
    total_revenue,
)


class SalesMetricsTest(unittest.TestCase):
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
                    "quantidade": 5,
                    "preco_unitario": 3000,
                    "desconto": 0.0,
                    "faturamento": 15000,
                },
                {
                    "data": "2024-08-02",
                    "produto": "Mouse",
                    "categoria": "Acessorios",
                    "regiao": "Sul",
                    "vendedor": "Carlos",
                    "quantidade": 10,
                    "preco_unitario": 100,
                    "desconto": 0.0,
                    "faturamento": 1000,
                },
            ]
        )
        self.data["data"] = pd.to_datetime(self.data["data"])

    def test_total_revenue(self):
        self.assertEqual(total_revenue(self.data), 46000)

    def test_total_quantity(self):
        self.assertEqual(total_quantity(self.data), 25)

    def test_percentage_change(self):
        self.assertEqual(percentage_change(100, 80), -20)

    def test_compare_periods(self):
        result = compare_periods(self.data, "2024-07", "2024-08")
        self.assertEqual(result["faturamento_anterior"], 30000)
        self.assertEqual(result["faturamento_atual"], 16000)
        self.assertAlmostEqual(result["variacao_percentual_faturamento"], -46.6666666667)
        self.assertIn("desconto_medio_anterior", result)
        self.assertIn("preco_unitario_medio_atual", result)

    def test_compare_periods_by_dimension(self):
        result = compare_periods_by_dimension(
            self.data, "2024-07", "2024-08", "produto"
        )
        notebook = result.loc[result["produto"] == "Notebook"].iloc[0]
        self.assertEqual(notebook["variacao_faturamento"], -15000)
        self.assertIn("variacao_desconto_medio", notebook)


if __name__ == "__main__":
    unittest.main()
