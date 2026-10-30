from src.llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    def generate(self, system_prompt, user_prompt, context):
        comparison = context.get("period_comparison", {})
        drivers = context.get("main_drivers", [])
        structured_report = context.get("structured_report", {})
        probable_causes = structured_report.get("causas_provaveis", [])

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
                f"O faturamento passou de {previous_revenue:.2f} em {previous_period} "
                f"para {current_revenue:.2f} em {current_period}, uma variacao de "
                f"{revenue_change:.2f} ({self._format_pct(revenue_change_pct)})."
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
                lines.append(
                    f"- {dimension} '{value}' teve variacao de faturamento de "
                    f"{change:.2f} ({self._format_pct(pct)}), variacao de quantidade "
                    f"de {quantity_change:.0f} e variacao de desconto medio de "
                    f"{discount_change:.2f} ponto(s)."
                )

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
