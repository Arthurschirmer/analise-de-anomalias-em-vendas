import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.evaluation.real_data_validation import audit_olist_source, run_real_data_validation


def _fixture(root, second_month="2024-08"):
    pd.DataFrame([
        {"order_id": "order_a", "customer_id": "customer_private_a", "order_status": "delivered", "order_purchase_timestamp": "2024-07-01 10:00:00"},
        {"order_id": "order_b", "customer_id": "customer_private_b", "order_status": "delivered", "order_purchase_timestamp": f"{second_month}-01 11:00:00"},
        {"order_id": "order_c", "customer_id": "customer_private_a", "order_status": "canceled", "order_purchase_timestamp": "2024-08-03 11:00:00"},
    ]).to_csv(root / "olist_orders_dataset.csv", index=False)
    pd.DataFrame([
        {"order_id": "order_a", "order_item_id": 1, "product_id": "product_a", "seller_id": "seller_anonymous_a", "price": 100.0, "freight_value": 10.0},
        {"order_id": "order_a", "order_item_id": 2, "product_id": "product_a", "seller_id": "seller_anonymous_a", "price": 100.0, "freight_value": 10.0},
        {"order_id": "order_b", "order_item_id": 1, "product_id": "product_b", "seller_id": "seller_anonymous_b", "price": 50.0, "freight_value": 5.0},
        {"order_id": "order_c", "order_item_id": 1, "product_id": "product_a", "seller_id": "seller_anonymous_a", "price": 999.0, "freight_value": 30.0},
    ]).to_csv(root / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame([
        {"product_id": "product_a", "product_category_name": "eletronicos"},
        {"product_id": "product_b", "product_category_name": None},
    ]).to_csv(root / "olist_products_dataset.csv", index=False)
    pd.DataFrame([
        {"customer_id": "customer_private_a", "customer_state": "SP"},
        {"customer_id": "customer_private_b", "customer_state": "RS"},
    ]).to_csv(root / "olist_customers_dataset.csv", index=False)
    pd.DataFrame([
        {"product_category_name": "eletronicos", "product_category_name_english": "electronics"},
    ]).to_csv(root / "product_category_name_translation.csv", index=False)


class RealDataValidationTest(unittest.TestCase):
    def test_nonfinite_parameters_are_rejected_before_loading_data(self):
        for kwargs in ({"threshold_pct": float("nan")}, {"min_period_value": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "finitos"):
                run_real_data_validation(input_dir="not_a_dataset", **kwargs)

    def test_reconciles_delivered_items_and_exports_reproducible_offline_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            output = root / "validation"
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be called")):
                summary = run_real_data_validation(root, output, min_period_value=0)
            audit = summary["auditoria"]
            self.assertEqual(audit["contagens_brutas"]["orders"], 3)
            self.assertEqual(audit["contagens_brutas"]["items"], 4)
            self.assertEqual(audit["pedidos_entregues"], 2)
            self.assertEqual(audit["itens_entregues"], 3)
            self.assertEqual(audit["linhas_padronizadas"], 2)
            self.assertEqual(audit["quantidade_padronizada"], 3)
            self.assertEqual(audit["faturamento_padronizado"], 250)
            self.assertEqual(audit["diferenca_faturamento"], 0)
            self.assertTrue(all(audit["verificacoes"].values()))
            self.assertEqual(audit["categorias_ausentes"]["itens_entregues"], 1)
            self.assertEqual(audit["categorias_ausentes"]["faturamento"], 50)
            self.assertFalse(summary["ground_truth"]["metricas_acuracia_calculadas"])
            self.assertEqual(summary["modo"], "local_mock_sem_api")
            self.assertEqual(len(summary["investigacoes"]), 1)
            self.assertEqual(summary["investigacoes"][0]["resumo_comparativo"]["variacao_faturamento"], -150)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in audit["arquivos"]))
            for path in summary["outputs"].values():
                self.assertTrue(Path(path).is_file(), path)
            monthly = pd.read_csv(output / "monthly_summary.csv")
            self.assertEqual(monthly["pedidos_entregues"].tolist(), [1, 1])
            self.assertEqual(monthly["itens"].tolist(), [2, 1])
            self.assertEqual(monthly["faturamento"].tolist(), [200, 50])
            detectors = pd.read_csv(output / "detector_comparison.csv")
            self.assertEqual(set(detectors.escopo), {"integral", "selecionado"})
            self.assertEqual(len(detectors), 10)
            exported = pd.read_csv(output / "selected_standardized.csv")
            self.assertNotIn("customer_id", exported)
            self.assertNotIn("order_id", exported)
            public_text = (output / "report.md").read_text(encoding="utf-8") + (output / "summary.json").read_text(encoding="utf-8")
            self.assertNotIn("customer_private_", public_text)

    def test_declared_window_does_not_change_full_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            summary = run_real_data_validation(root, root / "validation", start_period="2024-08", end_period="2024-08", min_period_value=0, selection_note="Recorte declarado para conferir um mês.")
            self.assertEqual(summary["auditoria"]["faturamento_padronizado"], 250)
            self.assertEqual(summary["escopos"]["selecionado"]["faturamento"], 50)
            self.assertEqual(summary["parametros"]["inicio_solicitado"], "2024-08")
            self.assertEqual(summary["sinais_percentuais_selecionados"], [])
            self.assertEqual(summary["investigacoes"], [])
            self.assertEqual(summary["parametros"]["justificativa_recorte"], "Recorte declarado para conferir um mês.")

    def test_duplicate_join_key_fails_before_amplifying_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            products = pd.read_csv(root / "olist_products_dataset.csv")
            pd.concat([products, products.iloc[[0]]], ignore_index=True).to_csv(root / "olist_products_dataset.csv", index=False)
            with self.assertRaisesRegex(ValueError, "chaves duplicadas"):
                audit_olist_source(root)

    def test_invalid_numeric_and_orphan_items_fail_structural_checks(self):
        for change in ("negative_price", "orphan_product"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _fixture(root)
                items = pd.read_csv(root / "olist_order_items_dataset.csv")
                if change == "negative_price":
                    items.loc[0, "price"] = -1
                else:
                    items.loc[0, "product_id"] = "missing_product"
                items.to_csv(root / "olist_order_items_dataset.csv", index=False)
                with self.assertRaisesRegex(ValueError, "Falha estrutural"):
                    audit_olist_source(root)

    def test_missing_months_are_reported_and_not_compared_as_consecutive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, second_month="2024-09")
            summary = run_real_data_validation(root, root / "validation", min_period_value=0)
            self.assertEqual(summary["auditoria"]["periodos_sem_registros"], ["2024-08"])
            self.assertEqual(summary["sinais_percentuais_selecionados"], [])
            self.assertTrue(all(row["comparacoes_consecutivas"] == 0 for row in summary["detetores"]))
            monthly = pd.read_csv(root / "validation" / "monthly_summary.csv")
            self.assertEqual(monthly.periodo.tolist(), ["2024-07", "2024-09"])

    def test_chart_flags_respect_minimum_value_instead_of_percentage_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            summary = run_real_data_validation(root, root / "validation", min_period_value=1000)
            self.assertEqual(summary["sinais_percentuais_selecionados"], [])
            for key in ("monthly_full_chart", "monthly_selected_chart"):
                html = Path(summary["outputs"][key]).read_text(encoding="utf-8")
                self.assertNotRegex(html, r'"anomalia"\s*:\s*true')

    def test_json_remains_strict_when_zscore_is_infinite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, second_month="2024-09")
            orders_path = root / "olist_orders_dataset.csv"
            orders = pd.read_csv(orders_path)
            orders.loc[len(orders)] = ["order_equal", "customer_private_a", "delivered", "2024-08-01 10:00:00"]
            orders.to_csv(orders_path, index=False)
            items_path = root / "olist_order_items_dataset.csv"
            items = pd.read_csv(items_path)
            items.loc[len(items)] = ["order_equal", 1, "product_a", "seller_anonymous_a", 200.0, 10.0]
            items.to_csv(items_path, index=False)
            summary = run_real_data_validation(root, root / "validation", min_period_value=0, max_investigations=0)
            def reject_constant(value):
                self.fail(f"Non-finite constant in JSON: {value}")
            stored = json.loads(Path(summary["outputs"]["summary"]).read_text(encoding="utf-8"), parse_constant=reject_constant)
            self.assertEqual(stored["modo"], "local_mock_sem_api")
            detectors = pd.read_csv(summary["outputs"]["detector_comparison"])
            self.assertTrue(detectors["z_score"].dropna().map(lambda value: abs(value) != float("inf")).all())

    def test_invalid_period_and_empty_window_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "AAAA-MM"):
            run_real_data_validation(start_period="julho")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            with self.assertRaisesRegex(ValueError, "não possui vendas"):
                run_real_data_validation(root, root / "validation", start_period="2025-01")


if __name__ == "__main__":
    unittest.main()

