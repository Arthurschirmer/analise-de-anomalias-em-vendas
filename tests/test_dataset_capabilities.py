import unittest

import pandas as pd

from src.agent.conclusion_context import MAX_CONTEXT_BYTES, build_conclusion_context
from src.agent.investigation_agent import InvestigationAgent
from src.analysis.cause_ranker import rank_probable_causes
from src.data.capabilities import SALES_DIMENSIONS, available_dimensions
from src.evaluation.agent_evaluator import find_unsupported_numbers
from src.llm.mock_provider import MockLLMProvider


def source_metadata():
    return {
        "source_type": "online_retail", "currency": "GBP", "currency_symbol": "£",
        "category_available": False, "seller_available": False,
        "discount_available": False, "available_dimensions": ["produto", "regiao"],
    }


def example_data():
    data = pd.DataFrame([
        {"data": "2011-01-02", "produto": "001A", "categoria": "Não informado",
         "regiao": "United Kingdom", "vendedor": "Não informado", "quantidade": 20,
         "preco_unitario": 100.0, "desconto": 0.0, "faturamento": 2000.0},
        {"data": "2011-02-02", "produto": "001A", "categoria": "Não informado",
         "regiao": "United Kingdom", "vendedor": "Não informado", "quantidade": 10,
         "preco_unitario": 100.0, "desconto": 0.0, "faturamento": 1000.0},
    ])
    data["data"] = pd.to_datetime(data["data"])
    data.attrs["dataset_context"] = source_metadata()
    return data


class DatasetCapabilitiesTest(unittest.TestCase):
    def test_defaults_explicit_empty_and_flags_are_respected(self):
        self.assertEqual(available_dimensions({}), SALES_DIMENSIONS)
        self.assertEqual(available_dimensions(source_metadata()), ("produto", "regiao"))
        self.assertEqual(available_dimensions({"category_available": False}),
                         ("produto", "regiao", "vendedor"))
        self.assertEqual(available_dimensions({"available_dimensions": []}), ())
        self.assertEqual(available_dimensions({"available_dimensions": ["lucro", {}, "regiao"]}), ("regiao",))
        self.assertEqual(available_dimensions({"contexto_da_base": source_metadata()}), ("produto", "regiao"))

    def test_investigation_never_executes_or_ranks_unavailable_dimensions(self):
        log = InvestigationAgent(MockLLMProvider()).investigate("Compare 2011-01 e 2011-02", example_data())
        comparison_tools = [result for result in log.tool_results if result.name == "comparar_periodos_por_dimensao"]
        self.assertEqual([result.parameters["dimensao"] for result in comparison_tools], ["produto", "regiao"])
        self.assertEqual({item["dimensao"] for item in log.evidence["main_drivers"]}, {"produto", "regiao"})
        self.assertEqual({item["dimensao"] for item in log.structured_report["causas_provaveis"]}, {"produto", "regiao"})
        self.assertEqual(log.evidence["period_comparison"]["variacao_faturamento"], -1000.0)
        self.assertIsNone(log.structured_report["resumo_comparativo"]["variacao_desconto_medio"])
        self.assertIn("£ 2000.00", log.final_conclusion)
        self.assertNotIn("R$", log.final_conclusion)
        self.assertNotIn("desconto medio", log.final_conclusion)
        self.assertEqual(find_unsupported_numbers(log), [])

    def test_summary_without_periods_also_avoids_unavailable_grouping(self):
        log = InvestigationAgent(MockLLMProvider()).investigate("Investigue esta base", example_data())
        names = {result.name for result in log.tool_results}
        self.assertIn("vendas_por_produto", names)
        self.assertIn("vendas_por_regiao", names)
        self.assertNotIn("vendas_por_categoria", names)
        self.assertNotIn("vendas_por_vendedor", names)

    def test_default_investigation_preserves_all_four_dimensions(self):
        data = example_data()
        data.attrs.clear()
        log = InvestigationAgent(MockLLMProvider()).investigate("Compare 2011-01 e 2011-02", data)
        self.assertEqual(
            [result.parameters["dimensao"] for result in log.tool_results if result.name == "comparar_periodos_por_dimensao"],
            list(SALES_DIMENSIONS),
        )
        self.assertIn("desconto medio", log.final_conclusion)

    def test_defensive_ranking_ignores_placeholder_groups_and_discount(self):
        log = InvestigationAgent(MockLLMProvider()).investigate("Compare 2011-01 e 2011-02", example_data())
        evidence = log.evidence
        original = dict(evidence["main_drivers"][0])
        unsupported = {**original, "dimensao": "categoria", "valor": "Não informado",
                       "variacao_faturamento": -10000.0, "variacao_desconto_medio": 0.99}
        evidence["main_drivers"].append(unsupported)
        for item in evidence["main_drivers"]:
            item["variacao_desconto_medio"] = 0.99
        causes = rank_probable_causes(evidence)
        self.assertNotIn("categoria", {item["dimensao"] for item in causes})
        self.assertTrue(all(item["metricas"]["variacao_desconto_medio"] is None for item in causes))
        self.assertTrue(all(not any("Desconto" in line for line in item["evidencias"]) for item in causes))

    def test_compact_context_keeps_currency_and_availability_within_budget(self):
        import json

        log = InvestigationAgent(MockLLMProvider()).investigate("Compare 2011-01 e 2011-02", example_data())
        log.evidence["main_drivers"].append({"dimensao": "vendedor", "valor": "Não informado"})
        context = build_conclusion_context(log.evidence, log.structured_report)
        encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.assertLessEqual(len(encoded), MAX_CONTEXT_BYTES)
        self.assertEqual(context["dataset_context"]["currency"], "GBP")
        self.assertEqual(context["dataset_context"]["currency_symbol"], "£")
        self.assertFalse(context["dataset_context"]["category_available"])
        self.assertFalse(context["dataset_context"]["seller_available"])
        self.assertFalse(context["dataset_context"]["discount_available"])
        self.assertEqual(context["dataset_context"]["available_dimensions"], ["produto", "regiao"])
        self.assertEqual({item["dimensao"] for item in context["main_drivers"]}, {"produto", "regiao"})

    def test_malformed_source_strings_cannot_expand_currency_or_availability_budget(self):
        import json

        log = InvestigationAgent(MockLLMProvider()).investigate("Compare 2011-01 e 2011-02", example_data())
        source = log.evidence["dataset_context"]
        for key in ("currency", "currency_symbol", "product_available", "region_available"):
            source[key] = "漢🚀" * 5000
        context = build_conclusion_context(log.evidence, log.structured_report)
        self.assertLessEqual(len(json.dumps(context, ensure_ascii=False).encode("utf-8")), MAX_CONTEXT_BYTES)
        self.assertLessEqual(len(context["dataset_context"]["currency"]), 40)
        self.assertLessEqual(len(context["dataset_context"]["currency_symbol"]), 40)
        self.assertNotIn("product_available", context["dataset_context"])
        self.assertNotIn("region_available", context["dataset_context"])
        self.assertFalse(context["dataset_context"]["category_available"])
        self.assertFalse(context["dataset_context"]["seller_available"])


if __name__ == "__main__":
    unittest.main()
