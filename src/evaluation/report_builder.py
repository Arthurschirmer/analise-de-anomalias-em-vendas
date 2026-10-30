from src.analysis.cause_ranker import rank_probable_causes


def build_structured_report(anomaly_description, evidence, tool_results):
    comparison = evidence.get("period_comparison", {})
    probable_causes = rank_probable_causes(evidence)

    report = {
        "anomalia": anomaly_description,
        "tipo_anomalia": _infer_anomaly_type(comparison),
        "periodos": _build_period_section(comparison),
        "resumo_comparativo": _build_summary_section(comparison),
        "causas_provaveis": probable_causes,
        "ferramentas_utilizadas": [
            {
                "nome": tool_result.name,
                "parametros": tool_result.parameters,
            }
            for tool_result in tool_results
        ],
        "limitacoes": _build_limitations(evidence, probable_causes),
    }

    if "automatic_detection" in evidence:
        report["deteccao_automatica"] = evidence["automatic_detection"]
    for key, label in [("revenue_decomposition", "decomposicao_faturamento"),
                       ("operational_indicators", "indicadores_operacionais")]:
        if key in evidence:
            report[label] = evidence[key]
            report["limitacoes"].extend(evidence[key].get("limitations", []))
    if evidence.get("dataset_context"):
        report["contexto_da_base"] = dict(evidence["dataset_context"])

    return report


def _build_period_section(comparison):
    if not comparison:
        return {}

    return {
        "periodo_anterior": comparison.get("periodo_anterior"),
        "periodo_atual": comparison.get("periodo_atual"),
    }


def _build_summary_section(comparison):
    if not comparison:
        return {}

    return {
        "faturamento_anterior": comparison.get("faturamento_anterior"),
        "faturamento_atual": comparison.get("faturamento_atual"),
        "variacao_faturamento": comparison.get("variacao_faturamento"),
        "variacao_percentual_faturamento": comparison.get(
            "variacao_percentual_faturamento"
        ),
        "quantidade_anterior": comparison.get("quantidade_anterior"),
        "quantidade_atual": comparison.get("quantidade_atual"),
        "variacao_quantidade": comparison.get("variacao_quantidade"),
        "variacao_percentual_quantidade": comparison.get(
            "variacao_percentual_quantidade"
        ),
        "variacao_desconto_medio": comparison.get("variacao_desconto_medio"),
        "variacao_preco_unitario_medio": comparison.get(
            "variacao_preco_unitario_medio"
        ),
    }


def _infer_anomaly_type(comparison):
    if not comparison:
        return "nao_identificada"

    revenue_change = comparison.get("variacao_faturamento", 0)
    if revenue_change < 0:
        return "queda_faturamento"
    if revenue_change > 0:
        return "aumento_faturamento"
    return "sem_variacao_faturamento"


def _build_limitations(evidence, probable_causes):
    limitations = list(evidence.get("dataset_context", {}).get("limitations", []))

    if not evidence.get("period_comparison"):
        limitations.append(
            "Nao foi possivel comparar periodos porque a anomalia nao indicou dois periodos reconhecidos."
        )

    if not probable_causes:
        limitations.append(
            "Nenhuma causa provavel foi ranqueada porque nao houve evidencias suficientes nas dimensoes analisadas."
        )

    limitations.append(
        "O ranking indica associacoes nos dados, mas nao prova causalidade sem informacoes externas."
    )
    limitations.append(
        "A investigacao usa somente as colunas disponiveis no CSV carregado."
    )

    return limitations
