from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from src.analysis.olist_operations import compare_olist_operations


class OlistOperationsTest(unittest.TestCase):
    def test_denominators_cancellation_and_date_deadline(self):
        with TemporaryDirectory() as folder:
            pd.DataFrame([
                ["A", "delivered", "2017-01-01", "2017-01-10", "2017-01-10 20:00:00"],
                ["B", "delivered", "2017-01-01", "2017-01-10", "2017-01-11"],
                ["C", "canceled", "2017-01-01", "2017-01-10", None],
                ["D", "delivered", "2017-01-01", None, "2017-01-11"],
                ["E", "delivered", "2017-02-01", "2017-02-10", "2017-02-09"],
                ["F", "shipped", "2017-02-01", "2017-02-10", None],
            ], columns=["order_id", "order_status", "order_purchase_timestamp",
                        "order_estimated_delivery_date", "order_delivered_customer_date"]).to_csv(
                            Path(folder, "olist_orders_dataset.csv"), index=False)
            result = compare_olist_operations(folder, "2017-01", "2017-02")
            self.assertTrue(result["available"])
            old, new = result["monthly"]
            self.assertEqual(old["taxa_cancelamento"], 25)
            self.assertEqual(old["entregas_avaliaveis"], 2)
            self.assertEqual(old["taxa_atraso_observada"], 50)
            self.assertEqual(new["taxa_atraso_observada"], 0)
            self.assertEqual(result["changes"]["taxa_atraso_observada_pp"], -50)
            self.assertEqual(new["pedidos_fora_denominador_atraso"], 1)
            self.assertNotIn("order_id", str(result))

    def test_missing_file_unavailable(self):
        with TemporaryDirectory() as folder:
            self.assertFalse(compare_olist_operations(folder, "2017-01", "2017-02")["available"])

