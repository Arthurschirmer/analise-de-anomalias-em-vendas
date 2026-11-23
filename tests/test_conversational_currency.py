import unittest

import pandas as pd

from src.agent.conversational_analytics import answer_sales_question, build_entity_index


def sample_sales(public_source=True):
    frame = pd.DataFrame({
        "data": pd.to_datetime(["2024-07-01", "2024-07-02", "2024-08-01", "2024-08-02"]),
        "produto": ["A", "B", "A", "B"],
        "categoria": ["Não informado"] * 4,
        "regiao": ["United Kingdom", "France", "United Kingdom", "France"],
        "vendedor": ["Não informado"] * 4,
        "quantidade": [2, 1, 1, 1],
        "preco_unitario": [10.0, 30.0, 10.0, 30.0],
        "desconto": [0.0] * 4,
        "faturamento": [20.0, 30.0, 10.0, 30.0],
    })
    if public_source:
        frame.attrs["dataset_context"] = {
            "source_type": "online_retail", "currency": "GBP", "currency_symbol": "£",
            "available_dimensions": ["produto", "regiao"],
            "category_available": False, "seller_available": False, "discount_available": False,
        }
    return frame


class ConversationalCurrencyTest(unittest.TestCase):
    def test_summary_and_unit_price_use_source_currency(self):
        data = sample_sales()
        for question, expected in (("Quanto faturei?", "£ 90,00"), ("Qual o preço médio?", "£ 18,00")):
            with self.subTest(question=question):
                result = answer_sales_question(question, data)
                self.assertIn(expected, result["content"])
                self.assertNotIn("R$", result["content"])
        self.assertEqual(data["faturamento"].sum(), 90.0)

    def test_comparison_uses_original_currency_for_every_amount(self):
        result = answer_sales_question("Compare julho de 2024 e agosto de 2024.", sample_sales())
        self.assertIn("£ 50,00", result["content"])
        self.assertIn("£ 40,00", result["content"])
        self.assertIn("£ -10,00", result["content"])
        self.assertNotIn("R$", result["content"])

    def test_country_alias_and_table_title_identify_unit(self):
        data = sample_sales()
        result = answer_sales_question("Mostre o faturamento por país.", data)
        self.assertIn("£", result["tables"][0]["title"])
        values = {row["Dimensao"]: row["Valor"] for row in result["tables"][0]["data"]}
        self.assertEqual(values, {"United Kingdom": 30.0, "France": 60.0})

    def test_unavailable_fields_are_not_indexed_or_returned_as_real_groups(self):
        data = sample_sales()
        index = build_entity_index(data)
        self.assertEqual(index["categoria"], [])
        self.assertEqual(index["vendedor"], [])
        for question in ("Faturamento por categoria", "Quais vendedores venderam mais?", "Qual o desconto?", "Mostre os descontos"):
            with self.subTest(question=question):
                result = answer_sales_question(question, data, entity_index=index)
                self.assertIn("não informa", result["content"])
                self.assertNotIn("tables", result)

    def test_remote_plans_cannot_use_placeholder_dimensions_or_discount(self):
        data = sample_sales()
        index = build_entity_index(data)
        plans = [
            {"operation": "ranking", "metric": "faturamento", "dimension": "categoria"},
            {"operation": "summary", "metric": "faturamento", "filters": {"vendedor": ["Não informado"]}},
            {"operation": "summary", "metric": "desconto"},
        ]
        for plan in plans:
            with self.subTest(plan=plan):
                result = answer_sales_question("Quanto faturei?", data, entity_index=index, query_plan=plan)
                self.assertIn("não informa", result["content"])
                self.assertNotIn("tables", result)

    def test_legacy_dataset_keeps_brl_and_available_dimensions(self):
        data = sample_sales(public_source=False)
        result = answer_sales_question("Quanto faturei?", data)
        self.assertIn("R$ 90,00", result["content"])
        self.assertEqual(build_entity_index(data)["categoria"], [("nao informado", "Não informado")])
        discount = answer_sales_question("Qual o desconto?", data)
        self.assertNotIn("não informa", discount["content"])


if __name__ == "__main__":
    unittest.main()
