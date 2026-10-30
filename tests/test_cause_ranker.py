import unittest

from src.analysis.cause_ranker import rank_probable_causes


class CauseRankerTest(unittest.TestCase):
    def test_rank_probable_causes_prioritizes_largest_drop(self):
        evidence = {
            "period_comparison": {
                "variacao_faturamento": -1000,
            },
            "main_drivers": [
                {
                    "dimensao": "produto",
                    "valor": "Notebook",
                    "faturamento_anterior": 1200,
                    "faturamento_atual": 200,
                    "variacao_faturamento": -1000,
                    "variacao_percentual_faturamento": -83.33,
                    "variacao_quantidade": -5,
                    "variacao_desconto_medio": 0.1,
                    "variacao_preco_unitario_medio": 0,
                },
                {
                    "dimensao": "regiao",
                    "valor": "Sul",
                    "faturamento_anterior": 800,
                    "faturamento_atual": 500,
                    "variacao_faturamento": -300,
                    "variacao_percentual_faturamento": -37.5,
                    "variacao_quantidade": -1,
                    "variacao_desconto_medio": 0,
                    "variacao_preco_unitario_medio": 0,
                },
            ],
        }

        ranked = rank_probable_causes(evidence)

        self.assertEqual(ranked[0]["valor"], "Notebook")
        self.assertEqual(ranked[0]["posicao"], 1)
        self.assertIn("hipotese", ranked[0])

    def test_rank_probable_causes_uses_correct_article_for_increase(self):
        evidence = {
            "period_comparison": {
                "variacao_faturamento": 1000,
            },
            "main_drivers": [
                {
                    "dimensao": "produto",
                    "valor": "Notebook",
                    "faturamento_anterior": 1000,
                    "faturamento_atual": 2000,
                    "variacao_faturamento": 1000,
                    "variacao_percentual_faturamento": 100,
                    "variacao_quantidade": 5,
                    "variacao_desconto_medio": 0,
                    "variacao_preco_unitario_medio": 0,
                }
            ],
        }

        ranked = rank_probable_causes(evidence)

        self.assertIn("O aumento pode estar relacionado", ranked[0]["hipotese"])


if __name__ == "__main__":
    unittest.main()
