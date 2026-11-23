import re
import unicodedata

import pandas as pd

from src.agent.period_parser import extract_periods_from_text
from src.analysis.sales_metrics import filter_by_period, percentage_change
from src.data.capabilities import available_dimensions
from src.formatting import currency_symbol, format_currency


DIMENSIONS = {
    "produto": ("produto", "produtos"),
    "categoria": ("categoria", "categorias"),
    "regiao": ("regiao", "regioes", "estado", "estados", "pais", "paises"),
    "vendedor": ("vendedor", "vendedores", "seller", "sellers"),
    "periodo": ("mes", "mensal", "periodo", "periodos"),
}

METRIC_LABELS = {
    "faturamento": "Faturamento",
    "quantidade": "Quantidade vendida",
    "preco_unitario": "Preco unitario medio",
    "desconto": "Desconto medio",
}

DIMENSION_LABELS = {
    "produto": "Produto",
    "categoria": "Categoria",
    "regiao": "Regiao",
    "vendedor": "Vendedor",
    "periodo": "Periodo",
}

ENTITY_DIMENSIONS = ("produto", "categoria", "regiao", "vendedor")


def normalize_text(value):
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def build_entity_index(data):
    """Pre-index dimension values so chat questions can filter without rescanning the CSV."""
    index = {}
    supported_dimensions = set(available_dimensions(data))
    for dimension in ENTITY_DIMENSIONS:
        if dimension not in supported_dimensions or dimension not in data.columns:
            index[dimension] = []
            continue

        values = data[dimension].dropna().astype(str).unique()
        normalized_values = [
            (normalize_text(value), value)
            for value in values
            if normalize_text(value) and normalize_text(value) not in {"nan", "none"}
        ]
        index[dimension] = sorted(normalized_values, key=lambda item: len(item[0]), reverse=True)
    return index


def answer_sales_question(question, data, entity_index=None, query_plan=None):
    """Answer a supported sales question using calculations over the loaded dataset."""
    if data.empty:
        return {"content": "A base carregada não contém registros para analisar."}

    normalized = normalize_text(question)
    metric, unsupported_metric = _detect_metric(normalized)
    unavailable = _unavailable_request(normalized, query_plan, data)
    if unavailable:
        return {"content": unavailable}
    if unsupported_metric:
        return {
            "content": (
                f"A base atual não tem uma coluna para calcular {unsupported_metric}. "
                "Posso analisar faturamento, quantidade, preço unitário médio ou desconto médio."
            )
        }

    entity_index = entity_index or {}
    dimension = _detect_dimension(normalized)
    filters = _find_entity_filters(normalized, entity_index)
    periods = extract_periods_from_text(question, data)
    plan = _validate_query_plan(query_plan, data, entity_index)
    if plan:
        if plan.get("operation") == "clarify":
            return {"content": plan.get("message") or "Pode especificar melhor o que deseja analisar?"}
        if plan.get("operation") == "unsupported":
            return {"content": plan.get("message") or "Essa pergunta pede uma metrica que a base nao permite calcular."}
        metric = plan.get("metric", metric)
        dimension = plan.get("dimension", dimension)
        periods = plan.get("periods", periods)
        filters.update(plan.get("filters", {}))

    if plan:
        is_comparison = plan.get("operation") == "compare"
        is_ranking = plan.get("operation") == "ranking"
    else:
        is_comparison = len(periods) >= 2
        is_ranking = _asks_for_ranking(normalized)
    if plan and plan.get("operation") == "compare" and len(periods) < 2:
        return {"content": "Quais dois periodos voce quer comparar?"}
    if plan and plan.get("operation") == "trend":
        dimension = "periodo"
    has_analytic_intent = _has_analytic_intent(normalized, dimension, filters, periods)

    if not has_analytic_intent:
        return None

    if is_comparison:
        return _answer_comparison(data, periods[:2], metric, dimension, filters)

    selected_period = periods[0] if len(periods) == 1 else None
    filtered_data = data
    if selected_period:
        filtered_data = filter_by_period(filtered_data, selected_period)
    elif plan and len(periods) > 1:
        period_values = filtered_data["data"].dt.to_period("M").astype(str)
        filtered_data = filtered_data.loc[period_values.isin(periods)]
    filtered_data = _apply_entity_filters(filtered_data, filters)

    if filtered_data.empty:
        return {
            "content": "Não encontrei registros para os períodos e filtros mencionados. Confira os nomes e as datas da base."
        }

    if is_ranking and dimension is None:
        return {
            "content": (
                "Para montar esse ranking, diga por qual dimensão quer ordenar: "
                "produto, categoria, região, vendedor ou mês."
            )
        }

    if dimension:
        table = _aggregate_by_dimension(filtered_data, metric, dimension)
        if is_ranking:
            descending = not _asks_for_smallest(normalized)
            table = table.sort_values("Valor", ascending=not descending)
            limit = plan.get("limit", _requested_limit(normalized)) if plan else _requested_limit(normalized)
            table = table.head(limit)
            direction = "maiores" if descending else "menores"
            title = f"Top {limit} por {DIMENSION_LABELS[dimension].lower()} ({direction} {_metric_label(metric, data).lower()})"
            content = f"Aqui estão os {direction} resultados por {DIMENSION_LABELS[dimension].lower()}:"
        else:
            table = table.sort_values("Valor", ascending=False)
            title = f"{_metric_label(metric, data)} por {DIMENSION_LABELS[dimension].lower()}"
            content = f"Resumo de {METRIC_LABELS[metric].lower()} por {DIMENSION_LABELS[dimension].lower()}:"

        context = _build_context(selected_period, filters)
        if context:
            content = f"{content} {context}"
        chart = {
            "type": "line" if dimension == "periodo" else "bar",
            "x": "Dimensao",
            "y": "Valor",
        }
        return {
            "content": content,
            "tables": [{"title": title, "data": table.to_dict(orient="records"), "chart": chart}],
        }

    value = _aggregate_value(filtered_data, metric)
    context = _build_context(selected_period, filters)
    suffix = f" ({context})" if context else ""
    return {
        "content": f"{METRIC_LABELS[metric]}{suffix}: **{_format_metric(value, metric, data)}**."
    }


def _answer_comparison(data, periods, metric, dimension, filters):
    previous_period, current_period = periods
    previous_data = _apply_entity_filters(filter_by_period(data, previous_period), filters)
    current_data = _apply_entity_filters(filter_by_period(data, current_period), filters)

    if previous_data.empty or current_data.empty:
        missing_periods = []
        if previous_data.empty:
            missing_periods.append(previous_period)
        if current_data.empty:
            missing_periods.append(current_period)
        return {
            "content": (
                "Nao encontrei registros para "
                f"{', '.join(missing_periods)} com os filtros informados. "
                "Confira os periodos e os valores da base."
            )
        }

    if dimension:
        previous = _aggregate_by_dimension(previous_data, metric, dimension).rename(
            columns={"Valor": "Periodo anterior"}
        )
        current = _aggregate_by_dimension(current_data, metric, dimension).rename(
            columns={"Valor": "Periodo atual"}
        )
        comparison = previous.merge(current, on="Dimensao", how="outer").fillna(0)
        comparison["Variacao"] = comparison["Periodo atual"] - comparison["Periodo anterior"]
        comparison["Variacao percentual"] = comparison.apply(
            lambda row: percentage_change(row["Periodo anterior"], row["Periodo atual"]),
            axis=1,
        )
        comparison["Variacao percentual"] = comparison["Variacao percentual"].map(
            lambda value: "-" if pd.isna(value) else round(float(value), 2)
        )
        overall_change = _aggregate_value(current_data, metric) - _aggregate_value(previous_data, metric)
        comparison = comparison.sort_values("Variacao", ascending=overall_change < 0)
        comparison = comparison.head(10)
        direction = "aumentou" if overall_change > 0 else "caiu" if overall_change < 0 else "não variou"
        content = (
            f"O {METRIC_LABELS[metric].lower()} total {direction} de "
            f"{_format_metric(_aggregate_value(previous_data, metric), metric, data)} em {previous_period} "
            f"para {_format_metric(_aggregate_value(current_data, metric), metric, data)} em {current_period}. "
            f"Comparação por {DIMENSION_LABELS[dimension].lower()}:"
        )
        context = _build_context(None, filters)
        if context:
            content = f"{content} Filtro: {context}."
        return {
            "content": content,
            "tables": [{
                "title": f"{_metric_label(metric, data)} por {DIMENSION_LABELS[dimension].lower()}",
                "data": comparison.to_dict(orient="records"),
            }],
        }

    previous_value = _aggregate_value(previous_data, metric)
    current_value = _aggregate_value(current_data, metric)
    change = current_value - previous_value
    change_pct = percentage_change(previous_value, current_value)
    pct_text = "não calculável, pois o período anterior é zero" if change_pct is None else f"{change_pct:.2f}%"
    return {
        "content": (
            f"Comparando {previous_period} com {current_period}, {METRIC_LABELS[metric].lower()} "
            f"passou de **{_format_metric(previous_value, metric, data)}** para "
            f"**{_format_metric(current_value, metric, data)}**. Variação: "
            f"**{_format_metric(change, metric, data)} ({pct_text})**."
        )
    }


def _aggregate_by_dimension(data, metric, dimension):
    if dimension == "periodo":
        working = data.assign(_dimension=data["data"].dt.to_period("M").astype(str))
    else:
        working = data.assign(_dimension=data[dimension].astype(str))

    rows = []
    for value, group in working.groupby("_dimension", dropna=False):
        rows.append({
            "Dimensao": value,
            "Valor": _aggregate_value(group, metric),
        })
    return pd.DataFrame(rows)


def _aggregate_value(data, metric):
    if data.empty:
        return 0.0
    if metric in {"faturamento", "quantidade"}:
        return float(data[metric].sum())
    if metric == "preco_unitario":
        quantity = float(data["quantidade"].sum())
        if quantity == 0:
            return 0.0
        return float((data["preco_unitario"] * data["quantidade"]).sum() / quantity)
    return float(data["desconto"].mean())


def _detect_metric(normalized):
    unsupported = {
        "lucro": "lucro",
        "margem": "margem de lucro",
        "custo": "custo",
        "ticket medio": "ticket médio por pedido",
    }
    for term, label in unsupported.items():
        if _contains(normalized, term):
            return "faturamento", label

    if any(_contains(normalized, term) for term in ("quantidade", "unidades", "unidade", "itens vendidos")):
        return "quantidade", None
    if any(_contains(normalized, term) for term in ("desconto", "descontos")):
        return "desconto", None
    if _contains(normalized, "preco"):
        return "preco_unitario", None
    if any(_contains(normalized, term) for term in ("faturamento", "receita", "faturou", "faturaram")):
        return "faturamento", None
    if any(
        _contains(normalized, term)
        for term in (
            "mais vendido", "mais vendidos", "mais vendida", "mais vendidas",
            "menos vendido", "menos vendidos", "menos vendida", "menos vendidas",
            "vendeu mais", "venderam mais", "vendeu menos", "venderam menos",
        )
    ):
        return "quantidade", None
    return "faturamento", None


def _detect_dimension(normalized):
    for dimension, terms in DIMENSIONS.items():
        if any(_contains(normalized, term) for term in terms):
            return dimension
    if any(_contains(normalized, term) for term in ("por mes", "ao longo do tempo", "evolucao", "tendencia", "grafico", "graficos", "plotar")):
        return "periodo"
    return None


def _find_entity_filters(normalized_question, entity_index):
    filters = {}
    for dimension in ENTITY_DIMENSIONS:
        matches = []
        for normalized_value, original_value in entity_index.get(dimension, []):
            if not normalized_value:
                continue
            if _contains(normalized_question, normalized_value):
                matches.append((normalized_value, original_value))

        # Prefer the full, more specific value when names overlap (for example, "Notebook Pro").
        selected = []
        for candidate in matches:
            if any(
                candidate[0] != other[0] and candidate[0] in other[0]
                for other in matches
            ):
                continue
            selected.append(candidate[1])
        if selected:
            filters[dimension] = list(dict.fromkeys(selected))
    return filters


def _validate_query_plan(plan, data, entity_index):
    if not isinstance(plan, dict):
        return None

    operation = plan.get("operation")
    allowed_operations = {"summary", "ranking", "compare", "trend", "clarify", "unsupported"}
    if operation not in allowed_operations:
        return None

    validated = {"operation": operation}
    if operation in {"clarify", "unsupported"}:
        validated["message"] = str(plan.get("message", ""))[:500]
        return validated

    metrics = set(METRIC_LABELS)
    metric = plan.get("metric")
    if metric in metrics:
        validated["metric"] = metric

    dimensions = set(DIMENSION_LABELS)
    dimension = plan.get("dimension")
    if dimension in dimensions:
        validated["dimension"] = dimension

    available_periods = set(data["data"].dt.to_period("M").astype(str).unique())
    requested_periods = plan.get("periods", [])
    if isinstance(requested_periods, list):
        normalized_periods = [
            period for period in requested_periods
            if isinstance(period, str) and re.fullmatch(r"\d{4}-\d{2}", period)
        ]
        if normalized_periods and any(period not in available_periods for period in normalized_periods):
            return {
                "operation": "clarify",
                "message": "Nao encontrei um dos periodos pedidos na base carregada. Confira as datas disponiveis.",
            }
        if normalized_periods:
            validated["periods"] = sorted(set(normalized_periods))

    validated_filters = {}
    requested_filters = plan.get("filters", {})
    if isinstance(requested_filters, dict):
        for dimension, requested_values in requested_filters.items():
            if dimension not in ENTITY_DIMENSIONS:
                continue
            if not isinstance(requested_values, list):
                requested_values = [requested_values]
            known_values = {
                normalized: original
                for normalized, original in entity_index.get(dimension, [])
            }
            matches = [
                known_values.get(normalize_text(value))
                for value in requested_values
                if isinstance(value, str)
            ]
            matches = [value for value in matches if value is not None]
            if matches:
                validated_filters[dimension] = list(dict.fromkeys(matches))
            elif requested_values:
                return {
                    "operation": "clarify",
                    "message": f"Nao encontrei o valor indicado para {DIMENSION_LABELS[dimension].lower()} na base.",
                }
    if validated_filters:
        validated["filters"] = validated_filters

    limit = plan.get("limit")
    if isinstance(limit, int):
        validated["limit"] = max(1, min(limit, 20))
    return validated


def _apply_entity_filters(data, filters):
    filtered = data
    for dimension, values in filters.items():
        if dimension in filtered.columns:
            filtered = filtered.loc[filtered[dimension].astype(str).isin(values)]
    return filtered


def _has_analytic_intent(normalized, dimension, filters, periods):
    terms = (
        "venda", "vendas", "vendeu", "venderam", "faturou", "faturamento",
        "receita", "quantidade", "unidade", "preco", "desconto", "total",
        "quanto", "grafico", "graficos", "plotar", "top", "ranking", "compare", "comparar", "comparacao",
        "variacao", "evolucao", "tendencia",
    )
    return bool(dimension or filters or periods or any(_contains(normalized, term) for term in terms))


def _asks_for_ranking(normalized):
    return any(
        _contains(normalized, term)
        for term in (
            "top", "ranking", "maior", "maiores", "menor", "menores",
            "melhor", "melhores", "pior", "piores", "mais vendeu",
            "mais venderam", "mais vendido", "mais vendidos", "mais vendida",
            "mais vendidas", "mais faturou", "mais faturaram", "vendeu mais",
            "venderam mais", "menos vendeu",
            "menos venderam", "menos vendidos", "menos vendidas", "vendeu menos",
            "venderam menos",
        )
    )


def _asks_for_smallest(normalized):
    return any(
        _contains(normalized, term)
        for term in ("menor", "menores", "pior", "piores", "menos vendeu", "menos venderam", "menos vendidos", "menos vendidas", "vendeu menos", "venderam menos")
    )


def _requested_limit(normalized):
    match = re.search(r"\btop\s+(\d{1,2})\b", normalized)
    if match:
        return max(1, min(int(match.group(1)), 20))
    return 5


def _build_context(period, filters):
    parts = []
    if period:
        parts.append(f"período {period}")
    for dimension, values in filters.items():
        label = DIMENSION_LABELS[dimension].lower()
        parts.append(f"{label}: {', '.join(values)}")
    return "; ".join(parts)


def _contains(text, term):
    start = 0
    while True:
        start = text.find(term, start)
        if start < 0:
            return False
        end = start + len(term)
        left_is_boundary = start == 0 or not _is_word_character(text[start - 1])
        right_is_boundary = end == len(text) or not _is_word_character(text[end])
        if left_is_boundary and right_is_boundary:
            return True
        start += 1


def _is_word_character(character):
    return character.isalnum() or character == "_"


def _unavailable_request(normalized, query_plan, data):
    metadata = data.attrs.get("dataset_context", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    plan = query_plan if isinstance(query_plan, dict) else {}
    requested_discount = (
        any(_contains(normalized, term) for term in ("desconto", "descontos"))
        or plan.get("metric") == "desconto"
    )
    if requested_discount and metadata.get("discount_available") is False:
        return (
            "Esta base não informa os descontos. O preenchimento zero não representa "
            "um desconto observado. Posso analisar faturamento, quantidade ou preço unitário."
        )
    supported = set(available_dimensions(data))
    filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
    for dimension in ENTITY_DIMENSIONS:
        requested = (
            any(_contains(normalized, term) for term in DIMENSIONS[dimension])
            or plan.get("dimension") == dimension
            or dimension in filters
        )
        if requested and dimension not in supported:
            return (
                f"Esta base não informa {DIMENSION_LABELS[dimension].lower()}. "
                "Esse campo está indisponível para agrupamentos e filtros; "
                "o marcador usado na importação não é uma observação real."
            )
    return None


def _metric_label(metric, context=None):
    label = METRIC_LABELS[metric]
    if metric in {"faturamento", "preco_unitario"} and currency_symbol(context) != "R$":
        return f"{label} ({currency_symbol(context)})"
    return label


def _format_metric(value, metric, context=None):
    if metric == "faturamento":
        return format_currency(value, context)
    if metric == "quantidade":
        return f"{value:,.0f}".replace(",", ".")
    if metric == "desconto":
        return f"{value:.2f}"
    return format_currency(value, context)
