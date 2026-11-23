from src.data.capabilities import available_dimensions, dataset_metadata


def rank_probable_causes(evidence, limit=5):
    comparison = evidence.get("period_comparison", {})
    dimensions = available_dimensions(evidence)
    drivers = [item for item in evidence.get("main_drivers", []) if item.get("dimensao") in dimensions]
    discount_available = dataset_metadata(evidence).get("discount_available") is not False

    if not comparison or not drivers:
        return []

    total_change = float(comparison.get("variacao_faturamento", 0.0))
    candidates = _filter_candidates_by_direction(drivers, total_change)

    max_revenue_change = _max_abs(candidates, "variacao_faturamento")
    max_quantity_change = _max_abs(candidates, "variacao_quantidade")
    max_discount_change = _max_abs(candidates, "variacao_desconto_medio") if discount_available else 0.0

    ranked = []
    for item in candidates:
        revenue_change = float(item.get("variacao_faturamento", 0.0))
        quantity_change = float(item.get("variacao_quantidade", 0.0))
        discount_change = float(item.get("variacao_desconto_medio", 0.0)) if discount_available else 0.0

        impact_score = _safe_ratio(abs(revenue_change), max_revenue_change)
        contribution_score = _safe_ratio(abs(revenue_change), abs(total_change))
        quantity_score = _safe_ratio(abs(quantity_change), max_quantity_change)
        discount_score = _safe_ratio(abs(discount_change), max_discount_change)

        base_score = (
            impact_score * 0.55
            + min(contribution_score, 1.0) * 0.25
            + quantity_score * 0.15
            + discount_score * 0.05
        ) * 100
        score = min(base_score * _dimension_weight(item["dimensao"]), 100.0)

        ranked.append(
            {
                "dimensao": item["dimensao"],
                "valor": item["valor"],
                "score": round(score, 2),
                "confianca": _confidence_label(score),
                "hipotese": _build_hypothesis(item, total_change),
                "evidencias": _build_evidence_list(item, total_change, discount_available),
                "metricas": {
                    "faturamento_anterior": item["faturamento_anterior"],
                    "faturamento_atual": item["faturamento_atual"],
                    "variacao_faturamento": revenue_change,
                    "variacao_percentual_faturamento": item[
                        "variacao_percentual_faturamento"
                    ],
                    "variacao_quantidade": quantity_change,
                    "variacao_desconto_medio": discount_change if discount_available else None,
                    "variacao_preco_unitario_medio": item.get(
                        "variacao_preco_unitario_medio", 0.0
                    ),
                },
            }
        )

    ranked = sorted(ranked, key=lambda item: item["score"], reverse=True)
    for index, item in enumerate(ranked[:limit], start=1):
        item["posicao"] = index
    return ranked[:limit]


def _filter_candidates_by_direction(drivers, total_change):
    if total_change < 0:
        candidates = [item for item in drivers if item["variacao_faturamento"] < 0]
    elif total_change > 0:
        candidates = [item for item in drivers if item["variacao_faturamento"] > 0]
    else:
        candidates = list(drivers)

    return candidates or list(drivers)


def _build_hypothesis(item, total_change):
    direction = "queda" if total_change < 0 else "aumento"
    article = "A" if direction == "queda" else "O"
    related = "relacionada" if direction == "queda" else "relacionado"
    dimension = item["dimensao"]
    value = item["valor"]
    return (
        f"{article} {direction} pode estar {related} a {dimension} '{value}', "
        "pois essa dimensao apresentou uma das maiores variacoes de faturamento."
    )


def _build_evidence_list(item, total_change, discount_available=True):
    direction = "queda" if total_change < 0 else "aumento"
    evidences = [
        (
            f"Faturamento teve {direction} de "
            f"{item['variacao_faturamento']:.2f} "
            f"({ _format_pct(item['variacao_percentual_faturamento']) })."
        ),
        f"Quantidade variou {item['variacao_quantidade']:.0f} unidade(s).",
    ]

    discount_change = item.get("variacao_desconto_medio", 0.0)
    if discount_available and abs(discount_change) > 0:
        evidences.append(
            f"Desconto medio variou {discount_change:.2f} ponto(s)."
        )

    price_change = item.get("variacao_preco_unitario_medio", 0.0)
    if abs(price_change) > 0:
        evidences.append(
            f"Preco unitario medio variou {price_change:.2f}."
        )

    return evidences


def _confidence_label(score):
    if score >= 70:
        return "alta"
    if score >= 40:
        return "media"
    return "baixa"


def _dimension_weight(dimension):
    weights = {
        "produto": 1.05,
        "regiao": 1.03,
        "vendedor": 1.02,
        "categoria": 0.96,
    }
    return weights.get(dimension, 1.0)


def _format_pct(value):
    if value is None:
        return "sem percentual calculavel"
    return f"{value:.2f}%"


def _max_abs(items, key):
    values = [abs(float(item.get(key, 0.0))) for item in items]
    return max(values) if values else 0.0


def _safe_ratio(value, max_value):
    if max_value == 0:
        return 0.0
    return value / max_value
