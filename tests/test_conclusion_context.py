import copy
import json
import unittest
from pathlib import Path

from src.agent.conclusion_context import (
    MAX_CONTEXT_BYTES, MAX_DESCRIPTION_CHARS, build_conclusion_context,
)
from src.agent.investigation_agent import InvestigationAgent, conclusion_request
from src.data.loader import load_sales_csv
from src.evaluation.llm_comparison import compare_investigation_reports
from src.llm.mock_provider import MockLLMProvider


ROOT = Path(__file__).resolve().parents[1]


def encoded(context):
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class CaptureProvider:
    def generate(self, system, user, context):
        self.request = (system, user, copy.deepcopy(context))
        return "Conclusao offline para verificacao de tamanho."


class ConclusionContextTest(unittest.TestCase):
    def test_real_demo_and_contextual_csv_keep_calculated_numbers_and_full_history(self):
        for path, before, after in (
            ("data/vendas_demo.csv", "2024-07", "2024-08"),
            ("data/contextual_demo/vendas.csv", "2024-01", "2024-02"),
        ):
            with self.subTest(path=path):
                capture = CaptureProvider()
                log = InvestigationAgent(capture).investigate(
                    f"Compare {before} e {after}", load_sales_csv(ROOT / path))
                context = capture.request[2]
                self.assertLessEqual(len(encoded(context)), MAX_CONTEXT_BYTES)
                self.assertEqual(context["period_comparison"]["faturamento_atual"],
                                 log.evidence["period_comparison"]["faturamento_atual"])
                self.assertEqual(context["revenue_decomposition"]["components"], [
                    {key: item[key] for key in ("key", "value", "available")}
                    for item in log.evidence["revenue_decomposition"]["components"]
                ])
                self.assertNotIn(b'"product_changes":', encoded(context))
                self.assertTrue(log.structured_report["decomposicao_faturamento"]["product_changes"])
                tool = next(item for item in log.tool_results if item.name == "decompor_faturamento")
                self.assertEqual(tool.result, log.evidence["revenue_decomposition"])
                # Compatibility includes formatting every retained numeric field.
                local = MockLLMProvider().generate(*capture.request)
                self.assertIn("faturamento passou", local)

    def _olist_evidence(self):
        log = InvestigationAgent(MockLLMProvider()).investigate(
            "Compare 2024-07 e 2024-08", load_sales_csv(ROOT / "data/vendas_demo.csv"))
        evidence, report = copy.deepcopy(log.evidence), copy.deepcopy(log.structured_report)
        evidence["dataset_context"] = {
            "source_type": "olist", "discount_available": False,
            "revenue_definition": "Preco dos itens sem frete, por data de compra.",
            "row_definition": "Itens agregados; linhas nao sao pedidos.",
            "quantity_definition": "Itens dos pedidos com os status selecionados.",
            "order_statuses": ["delivered"],
            "selected_start": "2017-01", "selected_end": "2018-07",
            "region_label": "UF do cliente", "price_definition": "Media afetada pelo mix.",
            "limitations": ["Desconto zero e preenchimento, nao uma observacao.",
                            "Nao permite lucro ou receita contabil."],
            "coverage": [{"month": "2017-01", "rows": 100}] * 1000,
            "raw_dir": "C:/private/source/files", "customer_ids": ["private-id"] * 500,
        }
        decomposition = evidence["revenue_decomposition"]
        decomposition["product_changes"] = [{"produto": "produto-confidencial", "rows": [1] * 1000}] * 50
        decomposition["product_changes_total"] = 5000
        evidence["operational_indicators"] = {
            "available": True,
            "monthly": [dict(periodo=period, pedidos=100, cancelados=10, taxa_cancelamento=10.0,
                             entregas_avaliaveis=60, entregas_atrasadas=6, taxa_atraso_observada=10.0,
                             pedidos_fora_denominador_atraso=40)
                        for period in ("2024-07", "2024-08")],
            "changes": {"taxa_cancelamento_pp": 0.0, "taxa_atraso_observada_pp": 0.0},
            "raw_orders": [{"order_id": "private-order"}] * 1000,
        }
        report["decomposicao_faturamento"] = decomposition
        report["indicadores_operacionais"] = evidence["operational_indicators"]
        return evidence, report

    def test_large_olist_details_stay_local_semantics_denominators_and_immutability(self):
        evidence, report = self._olist_evidence()
        original = copy.deepcopy((evidence, report))
        context = build_conclusion_context(evidence, report)
        payload = encoded(context)
        self.assertLessEqual(len(payload), MAX_CONTEXT_BYTES)
        self.assertEqual((evidence, report), original)
        self.assertFalse(context["dataset_context"]["discount_available"])
        for field in ("quantity_definition", "order_statuses", "selected_start", "selected_end"):
            self.assertEqual(context["dataset_context"][field], evidence["dataset_context"][field])
        self.assertEqual(context["operational_indicators"]["monthly"],
                         evidence["operational_indicators"]["monthly"])
        self.assertEqual(context["revenue_decomposition"]["product_changes_total"], 5000)
        for forbidden in ("private-id", "private-order", "produto-confidencial", "C:/private", '"coverage"'):
            self.assertNotIn(forbidden, payload.decode())
        constraints = " ".join(context["constraints"])
        for required in ("nao e lucro", "Descontos indisponiveis", "censura", "denominador", "causalidade",
                         "Mes sem registros nao e venda zero", "nao provam coleta incompleta"):
            self.assertIn(required, constraints)

    def test_multibyte_and_arbitrary_long_labels_use_explicit_unambiguous_abbreviations(self):
        evidence, report = self._olist_evidence()
        for index, driver in enumerate(evidence["main_drivers"]):
            driver["valor"] = "漢字🚀" * 5000 + str(index)
        for cause in report["causas_provaveis"]:
            cause["hipotese"] = "á🚀" * 5000
        for field in ("region_label", "revenue_definition", "row_definition", "price_definition"):
            evidence["dataset_context"][field] = "漢🚀" * 5000
        original = copy.deepcopy((evidence, report))
        _, user, context = conclusion_request("Pergunta🚀" * 5000, evidence, report)
        self.assertLessEqual(len(encoded(context)), MAX_CONTEXT_BYTES)
        self.assertEqual(json.loads(encoded(context)), context)
        self.assertLessEqual(len(user.split("Anomalia: ", 1)[1]), MAX_DESCRIPTION_CHARS)
        self.assertIn("[abreviado; ref:", user)
        self.assertEqual((evidence, report), original)
        labels = [item["valor"] for item in context.get("main_drivers", [])]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all("[abreviado; ref:" in value for value in labels))
        self.assertFalse(context["dataset_context"]["discount_available"])
        self.assertEqual(context["dataset_context"]["selected_start"], "2017-01")
        self.assertEqual(context["dataset_context"]["selected_end"], "2018-07")

    def test_status_scope_is_bounded_and_partial_list_is_explicit(self):
        evidence, report = self._olist_evidence()
        statuses = ["漢🚀" * 1000 + str(index) for index in range(9)]
        evidence["dataset_context"]["order_statuses"] = statuses
        context = build_conclusion_context(evidence, report)
        self.assertLessEqual(len(encoded(context)), MAX_CONTEXT_BYTES)
        retained = context["dataset_context"]["order_statuses"]
        self.assertEqual(len(retained), 8)
        self.assertTrue(all(len(value) <= 40 for value in retained))
        self.assertEqual(len(set(retained)), 8)
        self.assertEqual(context["context_meta"]["order_statuses_omitted"], 1)
        self.assertIn("Lista de status parcial", " ".join(context["constraints"]))
        self.assertEqual(evidence["dataset_context"]["order_statuses"], statuses)

    def test_pathological_integer_omits_whole_comparison_without_invalid_json(self):
        evidence = {"period_comparison": {"faturamento_atual": 10 ** 6000}}
        context = build_conclusion_context(evidence, {})
        self.assertLessEqual(len(encoded(context)), MAX_CONTEXT_BYTES)
        self.assertEqual(context["period_comparison"], {})
        self.assertIn("comparacao excede", " ".join(context["context_meta"]["omitted"]))

    def test_comparative_evaluation_receives_identical_shared_projection(self):
        capture = CaptureProvider()
        log = InvestigationAgent(capture).investigate(
            "Compare 2024-07 e 2024-08", load_sales_csv(ROOT / "data/vendas_demo.csv"))
        initial = capture.request
        result = compare_investigation_reports(log, provider=capture)
        self.assertEqual(capture.request, initial)
        self.assertEqual(result["context"], initial[2])
        self.assertLessEqual(len(encoded(result["context"])), MAX_CONTEXT_BYTES)


if __name__ == "__main__":
    unittest.main()
