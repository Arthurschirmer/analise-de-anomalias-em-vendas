import unittest
from types import SimpleNamespace

from src.evaluation.agent_evaluator import (
    evaluate_investigation,
    find_unsupported_numbers,
    summarize_evaluations,
)


class AgentEvaluatorTest(unittest.TestCase):
    def test_evaluate_investigation_marks_top1_hit(self):
        investigation = _fake_investigation("-100.00")
        scenario = {
            "id": "cenario_teste",
            "expected_cause": {"dimensao": "produto", "valor": "Notebook"},
            "expected_terms": ["Notebook", "desconto"],
        }

        evaluation = evaluate_investigation(investigation, scenario)

        self.assertTrue(evaluation["top1_match"])
        self.assertEqual(evaluation["expected_rank"], 1)
        self.assertIn("Notebook", evaluation["matched_terms"])
        self.assertEqual(evaluation["unsupported_numbers"], [])

    def test_find_unsupported_numbers_flags_numbers_outside_evidence(self):
        investigation = _fake_investigation("999.00")

        unsupported = find_unsupported_numbers(investigation)

        self.assertEqual(unsupported, ["999.00"])

    def test_summarize_evaluations_calculates_accuracy(self):
        evaluations = [
            {
                "top1_match": True,
                "top_k_match": True,
                "has_unsupported_numbers": False,
                "top_k": 3,
                "tempo_execucao_segundos": 0.1,
            },
            {
                "top1_match": False,
                "top_k_match": True,
                "has_unsupported_numbers": True,
                "top_k": 3,
                "tempo_execucao_segundos": 0.3,
            },
        ]

        summary = summarize_evaluations(evaluations)

        self.assertEqual(summary["total_cenarios"], 2)
        self.assertEqual(summary["taxa_acerto_top1_percentual"], 50.0)
        self.assertEqual(summary["taxa_acerto_top_k_percentual"], 100.0)
        self.assertEqual(summary["cenarios_com_numeros_nao_suportados"], 1)
        self.assertEqual(summary["tempo_medio_execucao_segundos"], 0.2)


def _fake_investigation(number_text):
    structured_report = {
        "causas_provaveis": [
            {
                "dimensao": "produto",
                "valor": "Notebook",
                "score": 90.0,
                "confianca": "alta",
                "metricas": {"variacao_faturamento": -100.0},
            }
        ],
        "resumo_comparativo": {"variacao_faturamento": -100.0},
    }
    evidence = {
        "main_drivers": [
            {
                "dimensao": "produto",
                "valor": "Notebook",
                "variacao_faturamento": -100.0,
                "variacao_desconto_medio": 0.2,
            }
        ]
    }
    conclusion = (
        "A queda pode estar relacionada ao produto Notebook, "
        f"com variacao de faturamento de {number_text}."
    )
    investigation = SimpleNamespace(
        structured_report=structured_report,
        evidence=evidence,
        final_conclusion=conclusion,
    )
    investigation.to_dict = lambda: {
        "structured_report": structured_report,
        "evidence": evidence,
        "final_conclusion": conclusion,
    }
    return investigation


if __name__ == "__main__":
    unittest.main()
