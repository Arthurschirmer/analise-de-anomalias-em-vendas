import unittest

import pandas as pd

from src.agent.alert_conversation import answer_alert_question, build_alert_context


class AlertConversationTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"data": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "produto": ["A", "B", "A"], "regiao": ["UK", "UK", "UK"], "quantidade": [2, 100, 500], "faturamento": [20., 1000., 5000.]})
        self.data.attrs["dataset_context"] = {"currency": "GBP", "source_label": "Loja", "available_dimensions": ["produto", "regiao"],
                                              "discount_available": False, "category_available": False, "seller_available": False}
        facts = {"period": "2024-01-02", "start_date": "2024-01-02", "end_date": "2024-01-02", "metric": "faturamento",
                 "observed": 20., "expected": 40., "difference": -20., "percentage": -50., "previous_observations": 60,
                 "score": 4., "threshold": 3., "calibration_method": "robust", "next_checks": ["Confira a cobertura."]}
        self.context = build_alert_context(self.data, facts, {"dimension": "produto", "segmento": "A"})

    def test_explanation_uses_selected_alert_and_source_currency(self):
        answer = answer_alert_question("Por que caiu?", self.data, self.context)
        self.assertIn("£ 20,00", answer["content"])
        self.assertIn("£ 40,00", answer["content"])
        self.assertIn("não confirmam", answer["content"])
        self.assertEqual(answer["mode"], "local")

    def test_quantities_and_groups_are_limited_to_day_and_segment(self):
        answer = answer_alert_question("Qual a quantidade?", self.data, self.context)
        self.assertIn("2,00 unidades", answer["content"])
        answer = answer_alert_question("Mostre os produtos", self.data, self.context)
        self.assertEqual(answer["tables"][0]["data"], [{"produto": "A", "faturamento": 20.}])

    def test_unknown_dimensions_and_causes_are_not_invented(self):
        for question in ("Qual vendedor?", "Qual categoria?", "Qual desconto?", "Qual lucro?"):
            self.assertIn("não informa", answer_alert_question(question, self.data, self.context)["content"])
        self.assertIn("não confirmam", answer_alert_question("Foi uma promoção?", self.data, self.context)["content"])
        self.context["calendar_context"] = "Promoção"
        self.assertIn("referência já considera", answer_alert_question("Foi uma promoção?", self.data, self.context)["content"])

    def test_other_dates_require_leaving_alert_context(self):
        for question in ("Investigue 2024-01-03", "Compare 2023-12 e 2024-01", "Qual a evolução mensal?", "Investigue fevereiro", "O que houve em 03/01/2024?"):
            self.assertIn("Sair do alerta", answer_alert_question(question, self.data, self.context)["content"])

    def test_rule_and_zero_reference_do_not_produce_fake_percentage(self):
        self.context.update(expected=0., percentage=None)
        answer = answer_alert_question("Qual a regra do escore?", self.data, self.context)["content"]
        self.assertIn("nula ou negativa", answer)
        self.assertIn("não é uma probabilidade", answer)

    def test_revenue_query_on_quantity_alert_keeps_metrics_separate(self):
        self.context.update(metric="quantidade", observed=2., expected=4., difference=-2.)
        answer = answer_alert_question("Qual o faturamento por produto?", self.data, self.context)
        self.assertEqual(answer["tables"][0]["data"], [{"produto": "A", "faturamento": 20.}])
        self.assertIn("£ 20,00", answer_alert_question("Qual o faturamento?", self.data, self.context)["content"])


if __name__ == "__main__":
    unittest.main()
