from src.agent.period_parser import extract_periods_from_text
from src.evaluation.investigation_log import InvestigationLog
from src.evaluation.report_builder import build_structured_report
from src.tools.sales_tools import SalesAnalysisTools


class InvestigationAgent:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def investigate(self, anomaly_description, data, extra_evidence=None):
        tools = SalesAnalysisTools(data)
        periods = self._get_periods(anomaly_description, data, extra_evidence)

        tools.run("faturamento_total")
        tools.run("quantidade_vendida")
        tools.run("vendas_por_periodo")

        evidence = {
            "anomaly_description": anomaly_description,
            "periods_detected": periods,
        }
        if data.attrs.get("dataset_context"):
            evidence["dataset_context"] = dict(data.attrs["dataset_context"])
        if extra_evidence:
            evidence["automatic_detection"] = extra_evidence

        if len(periods) >= 2:
            previous_period, current_period = periods[0], periods[1]
            comparison = tools.run(
                "comparar_periodos",
                periodo_anterior=previous_period,
                periodo_atual=current_period,
            )
            evidence["period_comparison"] = comparison.result
            evidence["revenue_decomposition"] = tools.run(
                "decompor_faturamento", periodo_anterior=previous_period,
                periodo_atual=current_period).result
            if data.attrs.get("olist_raw_dir"):
                evidence["operational_indicators"] = tools.run(
                    "indicadores_operacionais", periodo_anterior=previous_period,
                    periodo_atual=current_period).result

            drivers = []
            for dimension in ["produto", "categoria", "regiao", "vendedor"]:
                result = tools.run(
                    "comparar_periodos_por_dimensao",
                    periodo_anterior=previous_period,
                    periodo_atual=current_period,
                    dimensao=dimension,
                )
                drivers.extend(self._extract_main_drivers(result.result, dimension))

            total_change = comparison.result.get("variacao_faturamento", 0)
            evidence["main_drivers"] = sorted(
                drivers,
                key=lambda item: item["variacao_faturamento"],
                reverse=total_change > 0,
            )[:8]
        else:
            tools.run("vendas_por_produto")
            tools.run("vendas_por_categoria")
            tools.run("vendas_por_regiao")
            evidence["message"] = (
                "Nao foi possivel identificar dois periodos na descricao da anomalia."
            )

        structured_report = build_structured_report(
            anomaly_description,
            evidence,
            tools.history,
        )
        final_conclusion = self._generate_conclusion(
            anomaly_description,
            evidence,
            structured_report,
        )

        return InvestigationLog(
            anomaly_description=anomaly_description,
            final_conclusion=final_conclusion,
            tool_results=tools.history,
            evidence=evidence,
            structured_report=structured_report,
        )

    def _get_periods(self, anomaly_description, data, extra_evidence):
        if extra_evidence:
            previous_period = extra_evidence.get("periodo_anterior")
            current_period = extra_evidence.get("periodo_atual")
            if previous_period and current_period:
                return [previous_period, current_period]

        return extract_periods_from_text(anomaly_description, data)

    def _extract_main_drivers(self, comparison_data, dimension):
        records = comparison_data.to_dict(orient="records")
        drivers = []

        for record in records:
            drivers.append(
                {
                    "dimensao": dimension,
                    "valor": record[dimension],
                    "faturamento_anterior": float(record["faturamento_anterior"]),
                    "faturamento_atual": float(record["faturamento_atual"]),
                    "variacao_faturamento": float(record["variacao_faturamento"]),
                    "variacao_percentual_faturamento": (
                        None
                        if record["variacao_percentual_faturamento"] is None
                        else float(record["variacao_percentual_faturamento"])
                    ),
                    "variacao_quantidade": float(record["variacao_quantidade"]),
                    "desconto_medio_anterior": float(
                        record["desconto_medio_anterior"]
                    ),
                    "desconto_medio_atual": float(record["desconto_medio_atual"]),
                    "variacao_desconto_medio": float(
                        record["variacao_desconto_medio"]
                    ),
                    "preco_unitario_medio_anterior": float(
                        record["preco_unitario_medio_anterior"]
                    ),
                    "preco_unitario_medio_atual": float(
                        record["preco_unitario_medio_atual"]
                    ),
                    "variacao_preco_unitario_medio": float(
                        record["variacao_preco_unitario_medio"]
                    ),
                }
            )

        return drivers

    def _generate_conclusion(self, anomaly_description, evidence, structured_report):
        system_prompt, user_prompt, context = conclusion_request(anomaly_description, evidence, structured_report)
        return self.llm_provider.generate(system_prompt, user_prompt, context)


def conclusion_request(anomaly_description, evidence, structured_report):
    """Shared request used identically by investigations and explanation evaluation."""
    system_prompt = (
        "Voce e um agente de investigacao de anomalias em vendas. "
        "Use somente as evidencias fornecidas. Nao invente numeros, causas ou dados."
        " Diferencie associacao de causalidade e respeite as limitacoes da base. "
        "Campos preenchidos pelo adaptador nao sao observacoes reais."
        " Use a decomposicao de volume, mix e valor unitario quando disponivel; "
        "cancelamentos e atrasos usam coortes e denominadores distintos da receita."
    )
    user_prompt = (
        "Investigue a anomalia descrita e gere uma conclusao curta, baseada apenas nos resultados das ferramentas.\n"
        f"Anomalia: {anomaly_description}"
    )
    context = {**evidence, "structured_report": structured_report}
    return system_prompt, user_prompt, context
