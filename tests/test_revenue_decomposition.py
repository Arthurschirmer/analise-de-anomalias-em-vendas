import unittest
import pandas as pd
from src.analysis.revenue_decomposition import decompose_revenue_change


class RevenueDecompositionTest(unittest.TestCase):
    def data(self, before, after):
        return pd.DataFrame([
            {"data": pd.Timestamp(month + "-01"), "produto": product,
             "quantidade": qty, "faturamento": qty * price}
            for month, basket in [("2024-01", before), ("2024-02", after)]
            for product, (qty, price) in basket.items()])

    def test_reconciles_simultaneous_volume_mix_price_and_entries_exits(self):
        data = self.data({"A": (10, 10), "B": (20, 20), "C": (3, 30)},
                         {"A": (30, 12), "B": (10, 25), "D": (4, 15)})
        result = decompose_revenue_change(data, "2024-01", "2024-02")
        self.assertTrue(result["reconciled"])
        self.assertAlmostEqual(sum(x["value"] for x in result["components"]),
                               result["totals"]["variacao_faturamento"])
        self.assertEqual(result["component_values"]["entries"], 60)
        self.assertEqual(result["component_values"]["exits"], -90)
        self.assertNotEqual(result["component_values"]["mix"], 0)

    def test_price_only_does_not_become_volume_or_mix(self):
        data = self.data({"A": (10, 10), "B": (10, 20)}, {"A": (10, 15), "B": (10, 25)})
        effects = decompose_revenue_change(data, "2024-01", "2024-02")["component_values"]
        self.assertEqual(effects["volume"], 0)
        self.assertEqual(effects["mix"], 0)
        self.assertEqual(effects["unit_value"], 100)

    def test_disjoint_products_have_no_invented_prices(self):
        data = self.data({"A": (10, 10)}, {"B": (20, 20)})
        result = decompose_revenue_change(data, "2024-01", "2024-02")
        self.assertTrue(result["available"])
        self.assertFalse(result["common_basket_available"])
        self.assertTrue(all(x["valor_unitario_anterior"] is None or
                            x["valor_unitario_atual"] is None for x in result["product_changes"]))

    def test_invalid_quantities_and_missing_period_unavailable(self):
        data = self.data({"A": (0, 10)}, {"A": (20, 20)})
        self.assertFalse(decompose_revenue_change(data, "2024-01", "2024-02")["available"])
        self.assertFalse(decompose_revenue_change(data, "2023-12", "2024-02")["available"])

