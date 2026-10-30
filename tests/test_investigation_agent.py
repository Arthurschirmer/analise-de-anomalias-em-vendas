import unittest

import pandas as pd

from src.agent.investigation_agent import InvestigationAgent
from src.llm.mock_provider import MockLLMProvider


class InvestigationAgentTest(unittest.TestCase):
    def test_agent_uses_periods_from_automatic_detection(self):
        data = pd.DataFrame(
            [
                _row("2017-12-01", "Notebook", 10, 1000),
                _row("2018-01-01", "Notebook", 20, 2000),
            ]
        )
        data["data"] = pd.to_datetime(data["data"])
        agent = InvestigationAgent(MockLLMProvider())

        investigation = agent.investigate(
            "O faturamento aumentou em janeiro de 2018 em comparacao com dezembro de 2017.",
            data,
            extra_evidence={
                "periodo_anterior": "2017-12",
                "periodo_atual": "2018-01",
            },
        )

        self.assertEqual(
            investigation.evidence["periods_detected"],
            ["2017-12", "2018-01"],
        )
        self.assertEqual(
            investigation.evidence["period_comparison"]["periodo_anterior"],
            "2017-12",
        )
        self.assertEqual(
            investigation.evidence["period_comparison"]["periodo_atual"],
            "2018-01",
        )


def _row(date, product, quantity, revenue):
    return {
        "data": date,
        "produto": product,
        "categoria": "Eletronicos",
        "regiao": "SP",
        "vendedor": "seller_1",
        "quantidade": quantity,
        "preco_unitario": revenue / quantity,
        "desconto": 0.0,
        "faturamento": revenue,
    }


if __name__ == "__main__":
    unittest.main()
