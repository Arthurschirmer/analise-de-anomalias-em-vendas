import unittest

from src.evaluation.report_builder import build_structured_report


class FakeToolResult:
    name = "comparar_periodos"
    parameters = {"periodo_anterior": "2024-07", "periodo_atual": "2024-08"}


class ReportBuilderTest(unittest.TestCase):
    def test_dataset_semantics_are_included_in_report_limits(self):
        context = {
            "source_type": "olist", "discount_available": False,
            "limitations": ["Desconto não disponível; zero é preenchimento."],
        }
        report = build_structured_report("Análise real", {"dataset_context": context}, [])
        self.assertFalse(report["contexto_da_base"]["discount_available"])
        self.assertIn(context["limitations"][0], report["limitacoes"])

    def test_build_structured_report(self):
        evidence = {
            "period_comparison": {
                "periodo_anterior": "2024-07",
                "periodo_atual": "2024-08",
                "faturamento_anterior": 1000,
                "faturamento_atual": 600,
                "variacao_faturamento": -400,
                "variacao_percentual_faturamento": -40,
                "quantidade_anterior": 10,
                "quantidade_atual": 6,
                "variacao_quantidade": -4,
                "variacao_percentual_quantidade": -40,
                "variacao_desconto_medio": 0.05,
                "variacao_preco_unitario_medio": 0,
            },
            "main_drivers": [
                {
                    "dimensao": "produto",
                    "valor": "Notebook",
                    "faturamento_anterior": 1000,
                    "faturamento_atual": 600,
                    "variacao_faturamento": -400,
                    "variacao_percentual_faturamento": -40,
                    "variacao_quantidade": -4,
                    "variacao_desconto_medio": 0.05,
                    "variacao_preco_unitario_medio": 0,
                }
            ],
        }

        report = build_structured_report(
            "O faturamento caiu em agosto.",
            evidence,
            [FakeToolResult()],
        )

        self.assertEqual(report["tipo_anomalia"], "queda_faturamento")
        self.assertEqual(report["causas_provaveis"][0]["valor"], "Notebook")
        self.assertEqual(report["ferramentas_utilizadas"][0]["nome"], "comparar_periodos")


if __name__ == "__main__":
    unittest.main()
