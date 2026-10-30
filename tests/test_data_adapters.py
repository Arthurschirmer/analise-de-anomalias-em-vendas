import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.adapters.factory import load_sales_dataset
from src.data.adapters.olist_adapter import OlistAdapter


class DataAdaptersTest(unittest.TestCase):
    def test_load_csv_dataset_uses_standard_format(self):
        data = load_sales_dataset(dataset_type="csv", csv_path="data/vendas_demo.csv")

        self.assertIn("faturamento", data.columns)
        self.assertIn("produto", data.columns)
        self.assertGreater(len(data), 0)

    def test_olist_adapter_converts_raw_files_to_standard_format(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            _write_olist_fixture(raw_dir)

            data = OlistAdapter(raw_dir).load()

            self.assertEqual(len(data), 1)
            row = data.iloc[0]
            self.assertEqual(row["produto"], "produto_1")
            self.assertEqual(row["categoria"], "computers")
            self.assertEqual(row["regiao"], "RS")
            self.assertEqual(row["vendedor"], "seller_1")
            self.assertEqual(row["quantidade"], 2)
            self.assertEqual(row["faturamento"], 200.0)
            self.assertEqual(row["preco_unitario"], 100.0)
            self.assertEqual(row["desconto"], 0.0)
            self.assertFalse(data.attrs["dataset_context"]["discount_available"])
            self.assertEqual(data.attrs["dataset_context"]["region_label"], "UF do cliente")

    def test_olist_adapter_rejects_dimension_join_amplification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            _write_olist_fixture(raw_dir)
            products_path = raw_dir / "olist_products_dataset.csv"
            products = pd.read_csv(products_path)
            pd.concat([products, products], ignore_index=True).to_csv(products_path, index=False)
            with self.assertRaises(pd.errors.MergeError):
                OlistAdapter(raw_dir).load()

    def test_olist_adapter_requires_raw_directory(self):
        with self.assertRaises(FileNotFoundError):
            OlistAdapter("pasta_inexistente").load()


def _write_olist_fixture(raw_dir):
    pd.DataFrame(
        [
            {
                "order_id": "order_1",
                "customer_id": "customer_1",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-07-01 10:00:00",
            },
            {
                "order_id": "order_2",
                "customer_id": "customer_1",
                "order_status": "canceled",
                "order_purchase_timestamp": "2018-07-02 10:00:00",
            },
        ]
    ).to_csv(raw_dir / "olist_orders_dataset.csv", index=False)

    pd.DataFrame(
        [
            {
                "order_id": "order_1",
                "order_item_id": 1,
                "product_id": "produto_1",
                "seller_id": "seller_1",
                "shipping_limit_date": "2018-07-05 10:00:00",
                "price": 100.0,
                "freight_value": 20.0,
            },
            {
                "order_id": "order_1",
                "order_item_id": 2,
                "product_id": "produto_1",
                "seller_id": "seller_1",
                "shipping_limit_date": "2018-07-05 10:00:00",
                "price": 100.0,
                "freight_value": 20.0,
            },
            {
                "order_id": "order_2",
                "order_item_id": 1,
                "product_id": "produto_1",
                "seller_id": "seller_1",
                "shipping_limit_date": "2018-07-05 10:00:00",
                "price": 999.0,
                "freight_value": 20.0,
            },
        ]
    ).to_csv(raw_dir / "olist_order_items_dataset.csv", index=False)

    pd.DataFrame(
        [
            {
                "product_id": "produto_1",
                "product_category_name": "informatica_acessorios",
                "product_name_lenght": 10,
                "product_description_lenght": 20,
                "product_photos_qty": 1,
                "product_weight_g": 1000,
                "product_length_cm": 10,
                "product_height_cm": 10,
                "product_width_cm": 10,
            }
        ]
    ).to_csv(raw_dir / "olist_products_dataset.csv", index=False)

    pd.DataFrame(
        [
            {
                "customer_id": "customer_1",
                "customer_unique_id": "unique_1",
                "customer_zip_code_prefix": 90000,
                "customer_city": "ijui",
                "customer_state": "RS",
            }
        ]
    ).to_csv(raw_dir / "olist_customers_dataset.csv", index=False)

    pd.DataFrame(
        [
            {
                "product_category_name": "informatica_acessorios",
                "product_category_name_english": "computers",
            }
        ]
    ).to_csv(raw_dir / "product_category_name_translation.csv", index=False)


if __name__ == "__main__":
    unittest.main()
