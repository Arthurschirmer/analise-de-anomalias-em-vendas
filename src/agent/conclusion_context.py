"""Bounded evidence for explanations; the complete investigation stays local.

This is a projection, not another calculation. Numeric values that are retained
are copied unchanged, and no transaction rows or product-detail tables are sent.
"""

import hashlib
import json
import math
from numbers import Integral, Real

from src.data.capabilities import available_dimensions


MAX_CONTEXT_BYTES = 5500
MAX_DESCRIPTION_CHARS = 800

_COMPARISON_FIELDS = (
    "periodo_anterior", "periodo_atual", "faturamento_anterior", "faturamento_atual",
    "variacao_faturamento", "variacao_percentual_faturamento", "quantidade_anterior",
    "quantidade_atual", "variacao_quantidade", "variacao_percentual_quantidade",
    "variacao_desconto_medio", "variacao_preco_unitario_medio",
)
_DRIVER_FIELDS = (
    "dimensao", "valor", "variacao_faturamento", "variacao_percentual_faturamento",
    "variacao_quantidade", "variacao_desconto_medio", "variacao_preco_unitario_medio",
)
_SOURCE_FIELDS = (
    "source_type", "discount_available", "region_label", "product_label", "seller_label",
    "category_available", "seller_available", "product_available", "region_available",
    "currency", "currency_symbol",
    "revenue_definition", "quantity_definition", "row_definition", "price_definition",
    "selected_start", "selected_end",
    "comparison_window_definition",
)
_TOTAL_FIELDS = (
    "faturamento_anterior", "faturamento_atual", "variacao_faturamento",
    "quantidade_anterior", "quantidade_atual", "produtos_comuns",
    "entradas_observadas", "saidas_observadas", "soma_efeitos",
)
_OPERATION_FIELDS = (
    "periodo", "pedidos", "cancelados", "taxa_cancelamento", "entregas_avaliaveis",
    "entregas_atrasadas", "taxa_atraso_observada", "pedidos_fora_denominador_atraso",
)


def _shorten(value, limit):
    text = str(value)
    if len(text) <= limit:
        return text
    # The explicit marker and digest distinguish labels with the same prefix.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    marker = f"...[abreviado; ref:{digest}]"
    return text[:max(0, limit - len(marker))] + marker


def shorten_description(description):
    return _shorten(description, MAX_DESCRIPTION_CHARS)


def _select(source, fields, text_limit=120):
    """Copy only scalar fields; tables and unexpected nested objects stay local."""
    result = {}
    if not isinstance(source, dict):
        return result
    for key in fields:
        if key not in source:
            continue
        value = source[key]
        if value is None or isinstance(value, bool):
            result[key] = value
        elif isinstance(value, Integral):
            result[key] = int(value)
        elif isinstance(value, Real):
            result[key] = float(value) if math.isfinite(value) else None
        elif isinstance(value, str):
            result[key] = _shorten(value, text_limit)
    return result


def _rows(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _size(context):
    try:
        return len(json.dumps(context, ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))
    except (ValueError, OverflowError):
        # Python can reject the decimal representation of a pathological integer.
        # Treat it as over budget; a complete optional unit will be omitted.
        return math.inf


def _decomposition(source):
    result = _select(source, (
        "available", "reason", "common_basket_available", "product_changes_total",
        "reconciled", "reconciliation_error",
    ))
    result["totals"] = _select(source.get("totals", {}), _TOTAL_FIELDS)
    # Each effect appears once, with availability; there is no duplicate values map.
    result["components"] = [
        _select(item, ("key", "value", "available"), 40)
        for item in _rows(source.get("components", []))[:5]
    ]
    return result


def _operations(source, comparison):
    result = _select(source, ("available", "reason"))
    rows = _rows(source.get("monthly", []))
    periods = {comparison.get("periodo_anterior"), comparison.get("periodo_atual")}
    result["monthly"] = [
        _select(item, _OPERATION_FIELDS, 40)
        for item in rows if item.get("periodo") in periods
    ][:2]
    result["changes"] = _select(source.get("changes", {}), (
        "taxa_cancelamento_pp", "taxa_atraso_observada_pp",
    ))
    return result


def build_conclusion_context(evidence, structured_report):
    """Return valid compact JSON evidence within MAX_CONTEXT_BYTES UTF-8 bytes.

    Priority is comparison and source restrictions, accounting/operational
    aggregates, drivers, then ranking prose. Repeated sections and detail tables
    always stay in the original local report. Further removals are recorded.
    """
    source = evidence.get("dataset_context") or structured_report.get("contexto_da_base", {})
    comparison = _select(evidence.get("period_comparison", {}), _COMPARISON_FIELDS, 40)
    dimensions = available_dimensions(source)
    drivers = [item for item in _rows(evidence.get("main_drivers", [])) if item.get("dimensao") in dimensions]
    causes = [item for item in _rows(structured_report.get("causas_provaveis", [])) if item.get("dimensao") in dimensions]
    context = {
        "context_meta": {
            "version": 1,
            "full_report": "Relatorio e historico completos disponiveis localmente.",
            "omitted": ["linhas brutas, tabelas por produto, listas auxiliares e repeticoes"],
            "drivers_omitted": max(0, len(drivers) - 3),
            "causes_omitted": max(0, len(causes) - 3),
        },
        "constraints": [
            "Faturamento nao e lucro. Campos de preenchimento nao sao observacoes.",
            "Associacoes e ranking nao provam causalidade; score nao e probabilidade.",
            "Mes sem registros nao e venda zero; bordas/baixo volume nao provam coleta incompleta.",
            "Nomes/textos com [abreviado; ref:...] exigem consulta ao relatorio local; nao sao nomes completos.",
        ],
        "period_comparison": comparison,
        "dataset_context": _select(source, _SOURCE_FIELDS, 160),
        "main_drivers": [_select(item, _DRIVER_FIELDS, 96) for item in drivers[:3]],
        "structured_report": {"causas_provaveis": [
            _select(item, ("posicao", "hipotese", "score", "confianca", "dimensao", "valor"), 150)
            for item in causes[:3]
        ]},
    }
    # Source definitions are text/flags, never numeric measurements or records.
    context["dataset_context"] = {
        key: value for key, value in context["dataset_context"].items()
        if isinstance(value, (str, bool)) or value is None
    }
    for flag in ("category_available", "seller_available", "product_available", "region_available"):
        if flag in context["dataset_context"] and not isinstance(context["dataset_context"][flag], bool):
            del context["dataset_context"][flag]
    for field in ("currency", "currency_symbol"):
        if isinstance(context["dataset_context"].get(field), str):
            context["dataset_context"][field] = _shorten(source[field], 40)
    if isinstance(source, dict) and (
        isinstance(source.get("available_dimensions"), (list, tuple))
        or any(source.get(flag) is False for flag in (
            "category_available", "seller_available", "product_available", "region_available"
        ))
    ):
        # A canonical list is bounded to four known dimensions, never labels
        # copied from the dataset. Empty remains an explicit restriction.
        context["dataset_context"]["available_dimensions"] = list(available_dimensions(source))
        context["constraints"].append(
            "Investigue somente as dimensoes declaradas disponiveis; grupos de preenchimento nao sao evidencias."
        )
    if isinstance(source, dict) and source.get("currency"):
        context["constraints"].append(
            "Use a moeda declarada na fonte, sem conversao nem troca do simbolo ou unidade."
        )
    window = source.get("comparison_window", {}) if isinstance(source, dict) else {}
    if isinstance(window, dict) and window.get("mode") == "aligned_days":
        first, last = window.get("first_day"), window.get("last_day")
        if isinstance(first, int) and isinstance(last, int) and 1 <= first <= last <= 31:
            context["constraints"].append(f"Comparacao limitada aos dias {first} a {last} dos meses; nao descreva como meses completos.")
    for field in ("selected_start", "selected_end"):
        if isinstance(context["dataset_context"].get(field), str):
            context["dataset_context"][field] = _shorten(source[field], 40)
    statuses = source.get("order_statuses", []) if isinstance(source, dict) else []
    if isinstance(statuses, list):
        statuses = [value for value in statuses if isinstance(value, str)]
        if statuses:
            context["dataset_context"]["order_statuses"] = [
                _shorten(value, 40) for value in statuses[:8]
            ]
        if len(statuses) > 8:
            context["context_meta"]["order_statuses_omitted"] = len(statuses) - 8
            context["context_meta"]["omitted"].append("status adicionais da fonte; escopo parcial")
            context["constraints"].append(
                "Lista de status parcial: consulte o relatorio local antes de afirmar o escopo completo da receita."
            )
    limitations = source.get("limitations", []) if isinstance(source, dict) else []
    if isinstance(limitations, list):
        context["dataset_context"]["limitations"] = [
            _shorten(item, 150) for item in limitations[:3] if isinstance(item, str)
        ]
    if isinstance(source, dict) and source.get("discount_available") is False:
        context["constraints"].append(
            "Descontos indisponiveis: zeros de preenchimento nao permitem concluir ausencia ou efeito de desconto."
        )

    decomposition = evidence.get("revenue_decomposition") or structured_report.get("decomposicao_faturamento")
    if isinstance(decomposition, dict):
        context["revenue_decomposition"] = _decomposition(decomposition)
        context["constraints"].append(
            "Decomposicao contabil sequencial: volume, mix, valor unitario; efeitos dependem dessa ordem. "
            "Valor unitario efetivo nao e preco de tabela. Entradas/saidas sao presenca nos meses, nao ciclo de vida."
        )
    operations = evidence.get("operational_indicators") or structured_report.get("indicadores_operacionais")
    if isinstance(operations, dict):
        context["operational_indicators"] = _operations(operations, comparison)
        context["constraints"].append(
            "Operacoes usam coortes por compra/todos os status; diferem da receita. Cancelamentos sao status finais "
            "sem data de decisao. Atraso: so entregues com datas validas; pendentes e datas ausentes excluidos "
            "do denominador (censura). Associacao nao prova causa."
        )

    def fits():
        return _size(context) <= MAX_CONTEXT_BYTES

    def omitted(section):
        if section not in context["context_meta"]["omitted"]:
            context["context_meta"]["omitted"].append(section)

    # Drop complete optional units, never fragments of serialized JSON/numbers.
    ranking = context["structured_report"]["causas_provaveis"]
    while not fits() and ranking:
        ranking.pop()
        context["context_meta"]["causes_omitted"] += 1
    for field in ("limitations", "price_definition", "row_definition", "seller_label", "product_label"):
        if not fits() and field in context["dataset_context"]:
            del context["dataset_context"][field]
            omitted("detalhes adicionais da fonte por limite de tamanho")
    while not fits() and len(context["main_drivers"]) > 1:
        context["main_drivers"].pop()
        context["context_meta"]["drivers_omitted"] += 1
    if not fits() and "revenue_decomposition" in context:
        context["revenue_decomposition"].pop("totals", None)
        omitted("totais repetidos da decomposicao por limite de tamanho")
    for section in ("operational_indicators", "revenue_decomposition", "main_drivers"):
        if not fits() and section in context:
            if section == "main_drivers":
                context["context_meta"]["drivers_omitted"] += len(context[section])
            del context[section]
            omitted(section + " por limite de tamanho")
    for field in ("revenue_definition", "region_label", "source_type"):
        if not fits() and field in context["dataset_context"]:
            del context["dataset_context"][field]
            omitted("detalhes adicionais da fonte por limite de tamanho")
    if not fits():
        # Defensive fallback for pathological scalar inputs, e.g. enormous ints.
        # An incomplete comparison would make the deterministic provider invent
        # defaults or fail, so omit that comparison as a complete unit.
        context["period_comparison"] = {}
        omitted("comparacao excede o limite; nao concluir numeros sem consultar relatorio local")
    # Preserve selection dates and unavailable-discount flags even with unusually
    # long source text; the status list can be omitted only as an explicit unit.
    if not fits():
        context["dataset_context"].pop("quantity_definition", None)
        omitted("definicao da unidade por limite de tamanho")
    if not fits() and "order_statuses" in context["dataset_context"]:
        removed = context["dataset_context"].pop("order_statuses")
        context["context_meta"]["order_statuses_omitted"] = len(removed) + context["context_meta"].get("order_statuses_omitted", 0)
        omitted("status da fonte por limite de tamanho; consultar escopo completo localmente")
    return context
