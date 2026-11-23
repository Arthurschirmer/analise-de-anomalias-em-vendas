from src.llm.provider import LLMProvider
from src.data.capabilities import dataset_metadata
from src.formatting import currency_symbol


class MockLLMProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, context):
        comparison = context.get("period_comparison", {})
        drivers = context.get("main_drivers", [])
        structured_report = context.get("structured_report", {})
        probable_causes = structured_report.get("causas_provaveis", [])
        discount_available = dataset_metadata(context).get("discount_available") is not False

        previous_period = comparison.get("periodo_anterior")
        current_period = comparison.get("periodo_atual")
        previous_revenue = comparison.get("faturamento_anterior")
        current_revenue = comparison.get("faturamento_atual")
        revenue_change = comparison.get("variacao_faturamento")
        revenue_change_pct = comparison.get("variacao_percentual_faturamento")

        lines = [
            "A investigacao foi realizada em modo mock, usando apenas os resultados das ferramentas de analise.",
        ]

        if comparison:
            lines.append(
                f"O faturamento passou de {self._money(previous_revenue, context)} em {previous_period} "
                f"para {self._money(current_revenue, context)} em {current_period}, uma variacao de "
                f"{self._money(revenue_change, context)} ({self._format_pct(revenue_change_pct)})."
            )

        if drivers:
            lines.append("As principais evidencias encontradas foram:")
            for item in drivers[:5]:
                dimension = item["dimensao"]
                value = item["valor"]
                change = item["variacao_faturamento"]
                pct = item["variacao_percentual_faturamento"]
                quantity_change = item.get("variacao_quantidade", 0)
                discount_change = item.get("variacao_desconto_medio", 0)
                line = (
                    f"- {dimension} '{value}' teve variacao de faturamento de "
                    f"{self._money(change, context)} ({self._format_pct(pct)}), variacao de quantidade "
                    f"de {quantity_change:.0f}"
                )
                if discount_available:
                    line += f" e variacao de desconto medio de {discount_change:.2f} ponto(s)"
                lines.append(line + ".")

        if probable_causes:
            lines.append("Ranking de causas provaveis:")
            for cause in probable_causes[:3]:
                lines.append(
                    f"{cause['posicao']}. {cause['hipotese']} "
                    f"Score: {cause['score']:.2f}; confianca {cause['confianca']}."
                )

        lines.append(
            "Com base nessas evidencias, a conclusao prioriza as causas ranqueadas pelo maior impacto nos dados. Nenhum numero foi inferido fora dos dados calculados."
        )
        return "\n".join(lines)

    def _format_pct(self, value):
        if value is None:
            return "sem percentual calculavel"
        return f"{value:.2f}%"

    def _money(self, value, context):
        # Keep the legacy decimal form used by the scenario evaluator. Only the
        # unit changes; no exchange rate or grouping is applied to the number.
        symbol = currency_symbol(context)
        prefix = f"{symbol} " if symbol != "R$" else ""
        return f"{prefix}{value:.2f}"
