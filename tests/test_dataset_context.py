import json
import unittest

import pandas as pd

from src.ui.dataset_context import dataset_context


def example_data():
    return pd.DataFrame({
        "data": pd.to_datetime(["2024-01-15", "2024-01-31", "2024-03-01", "2024-03-29"]),
        "produto": ["id_a", "id_b", "id_a", "id_b"],
        "categoria": ["sem_categoria", None, "electronics", "furniture"],
        "regiao": ["SP", "RS", "SP", "RS"],
        "vendedor": ["id_v1", "id_v2", "id_v1", "id_v2"],
        "quantidade": [2, 3, 50, 60],
        "preco_unitario": [50.0, 50.0, 4.0, 5.0],
        "desconto": [0.0, 0.0, 0.0, 0.0],
        "faturamento": [100.0, 150.0, 200.0, 300.0],
    })


class DatasetContextTest(unittest.TestCase):
    def test_olist_semantics_distinguish_ids_ufs_and_placeholder_discount(self):
        context = dataset_context(example_data(), "olist")
        self.assertEqual(context["source_label"], "Base pública · Olist")
        self.assertEqual(context["region_label"], "UF do cliente")
        self.assertEqual(context["product_label"], "ID do produto")
        self.assertEqual(context["seller_label"], "ID do vendedor")
        self.assertFalse(context["discount_available"])
        self.assertIn("pedidos entregues", context["revenue_definition"])
        self.assertIn("frete excluído", context["revenue_definition"])
        self.assertIn("não representa necessariamente um pedido", context["row_definition"])
        constraints = " ".join(context["ai_context"]["restricoes"])
        self.assertIn("Não atribua variações a descontos", constraints)
        self.assertIn("não invente nomes", constraints)
        self.assertNotIn("id_a", json.dumps(context, ensure_ascii=False))

    def test_custom_csv_zero_discounts_do_not_become_missing_observations(self):
        data = example_data()
        context = dataset_context(data, "csv")
        self.assertTrue(context["discount_available"])
        self.assertEqual(context["region_label"], "Região")
        self.assertNotIn("preenchimento técnico", " ".join(context["notes"]))
        data.attrs["dataset_context"] = {"discount_available": False}
        context = dataset_context(data, "csv")
        self.assertFalse(context["discount_available"])
        self.assertIn("origem", " ".join(context["notes"]))

    def test_calendar_coverage_preserves_missing_months_and_exploratory_threshold(self):
        coverage = dataset_context(example_data(), "olist")["monthly_coverage"]
        self.assertEqual(coverage["first_date"], "2024-01-15")
        self.assertEqual(coverage["last_date"], "2024-03-29")
        self.assertEqual(coverage["observed_months"], 2)
        self.assertEqual(coverage["observed_periods"], ["2024-01", "2024-03"])
        self.assertEqual(coverage["missing_months"], ["2024-02"])
        self.assertEqual(coverage["sparse_threshold_quantity"], 100)
        self.assertEqual(coverage["sparse_months"], [
            {"periodo": "2024-01", "quantidade": 5.0, "registros": 2},
        ])
        self.assertEqual(coverage["possible_partial_months"], ["2024-01", "2024-03"])
        first, last = coverage["boundary_months"]
        self.assertTrue(first["starts_after_month_start"])
        self.assertFalse(first["ends_before_month_end"])
        self.assertFalse(last["starts_after_month_start"])
        self.assertTrue(last["ends_before_month_end"])
        notes = " ".join(dataset_context(example_data(), "olist")["notes"])
        self.assertIn("não demonstra que a coleta está incompleta", notes)
        self.assertIn("dias sem venda", notes)

    def test_missing_categories_are_measured_without_inventing_business_category(self):
        context = dataset_context(example_data(), "olist")
        self.assertEqual(context["missing_category"], {
            "row_count": 2, "units": 5.0, "revenue": 250.0,
        })
        self.assertIn("sem_categoria indica informação ausente", " ".join(context["ai_context"]["restricoes"]))

    def test_nested_adapter_metadata_is_respected_and_result_is_json_serializable(self):
        data = example_data()
        data.attrs["dataset_context"] = {
            "revenue_definition": "Valor dos itens dos pedidos com status escolhido; frete excluído.",
            "quantity_definition": "Itens do conjunto de pedidos selecionado.",
            "order_statuses": ["shipped"],
        }
        context = dataset_context(data, "olist")
        self.assertEqual(context["revenue_definition"], data.attrs["dataset_context"]["revenue_definition"])
        self.assertEqual(context["quantity_definition"], data.attrs["dataset_context"]["quantity_definition"])
        self.assertEqual(context["ai_context"]["status_pedidos_incluidos"], ["shipped"])
        serialized = json.dumps(context, ensure_ascii=False, allow_nan=False)
        self.assertIn("2024-02", serialized)
        data.attrs["dataset_context"] = {"order_statuses": ["shipped"]}
        custom = dataset_context(data, "olist")
        self.assertIn("status selecionados", custom["revenue_definition"])
        self.assertNotIn("entregues", custom["quantity_definition"])

    def test_empty_and_single_month_coverage_are_explicit(self):
        empty = dataset_context(pd.DataFrame(), "csv")
        self.assertEqual(empty["monthly_coverage"]["observed_months"], 0)
        self.assertEqual(empty["monthly_coverage"]["missing_months"], [])
        self.assertIsNone(empty["monthly_coverage"]["first_date"])
        single = dataset_context(example_data().iloc[:2].copy(), "demo")
        self.assertEqual(single["source_label"], "Dados fictícios")
        self.assertEqual(single["monthly_coverage"]["observed_months"], 1)
        self.assertEqual(len(single["monthly_coverage"]["boundary_months"]), 1)


if __name__ == "__main__":
    unittest.main()

