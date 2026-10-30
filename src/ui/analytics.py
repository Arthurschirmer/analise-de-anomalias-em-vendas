"""Deterministic data prepared for the anomaly dashboard.

No model or UI calls are made here. Percentages with a zero or absent baseline
remain unavailable; the interface must not display them as a zero change.
"""

import pandas as pd

from src.analysis.sales_metrics import compare_periods, percentage_change, sales_by_period


_MONTHS = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")
_METRIC_COLUMNS = {
    "faturamento": "faturamento",
    "quantidade": "quantidade",
    "preco_unitario": "preco_unitario_medio",
    "preco_unitario_medio": "preco_unitario_medio",
    "desconto": "desconto_medio",
    "desconto_medio": "desconto_medio",
}
_DIMENSIONS = {"produto", "categoria", "regiao", "vendedor"}
_MONTHLY_COLUMNS = [
    "periodo", "periodo_label", "valor", "valor_anterior", "variacao",
    "variacao_percentual", "comparavel", "percentual_disponivel", "anomalia", "direcao",
]
_COMPARISON_VALUES = [
    "faturamento_anterior", "faturamento_atual", "variacao_faturamento",
    "variacao_percentual_faturamento", "quantidade_anterior", "quantidade_atual",
    "variacao_quantidade", "variacao_percentual_quantidade", "desconto_medio_anterior",
    "desconto_medio_atual", "variacao_desconto_medio", "preco_unitario_medio_anterior",
    "preco_unitario_medio_atual", "variacao_preco_unitario_medio",
    "variacao_percentual_preco_unitario_medio", "variacao_percentual_desconto_medio",
    "registros_anterior", "registros_atual",
]
_CONTRIBUTION_COLUMNS = [
    "dimensao", "segmento", "faturamento_anterior", "faturamento_atual",
    "variacao_faturamento", "variacao_percentual_faturamento",
    "participacao_variacao_percentual", "quantidade_anterior", "quantidade_atual",
    "variacao_quantidade", "preco_unitario_medio_anterior", "preco_unitario_medio_atual",
    "variacao_preco_unitario_medio", "desconto_medio_anterior", "desconto_medio_atual",
    "variacao_desconto_medio", "direcao",
]


def period_label(period):
    """Return an abbreviated Portuguese month label, e.g. ``jul/2024``."""
    if period is None:
        return "—"
    try:
        value = pd.Period(period, freq="M")
        return f"{_MONTHS[value.month - 1]}/{value.year}"
    except (TypeError, ValueError):
        return str(period)


def _prepared_data(data):
    if data is None or data.empty or "data" not in data.columns:
        return pd.DataFrame()
    prepared = data.copy()
    prepared["data"] = pd.to_datetime(prepared["data"], errors="coerce")
    return prepared.loc[prepared["data"].notna()]


def available_periods(data):
    """Observed months only, sorted chronologically; absent months are not invented."""
    prepared = _prepared_data(data)
    if prepared.empty:
        return []
    return sorted(prepared["data"].dt.to_period("M").astype(str).unique().tolist())


def monthly_overview(data, metric="faturamento", threshold_pct=30.0):
    """Monthly values and flags from an absolute percentage-change threshold.

    A flag is a rule-based alert, not a statistical significance or confidence
    score. Only consecutive observed calendar months are compared. A comparison
    with a zero previous value has a known absolute change but no percentage.
    """
    if metric not in _METRIC_COLUMNS:
        raise ValueError(f"Métrica indisponível: {metric}")
    if float(threshold_pct) < 0:
        raise ValueError("O limite percentual deve ser maior ou igual a zero.")
    prepared = _prepared_data(data)
    if prepared.empty:
        return pd.DataFrame(columns=_MONTHLY_COLUMNS)

    grouped = sales_by_period(prepared).reset_index(drop=True)
    result = grouped[["periodo", _METRIC_COLUMNS[metric]]].rename(
        columns={_METRIC_COLUMNS[metric]: "valor"}
    )
    result["periodo_label"] = result["periodo"].map(period_label)
    periods = pd.PeriodIndex(result["periodo"], freq="M")
    consecutive = pd.Series(
        [False] + [periods[i] == periods[i - 1] + 1 for i in range(1, len(periods))],
        index=result.index,
    )
    result["valor_anterior"] = result["valor"].shift().where(consecutive)
    result["comparavel"] = consecutive
    result["variacao"] = (result["valor"] - result["valor_anterior"]).where(consecutive)
    result["percentual_disponivel"] = consecutive & result["valor_anterior"].ne(0)
    result["variacao_percentual"] = (
        result["variacao"].div(result["valor_anterior"].where(result["percentual_disponivel"])) * 100
    )
    result["anomalia"] = result["variacao_percentual"].abs().ge(float(threshold_pct)).fillna(False)
    result["direcao"] = result["variacao"].map(
        lambda value: None if pd.isna(value) else "queda" if value < 0 else "aumento" if value > 0 else "estavel"
    )
    return result[_MONTHLY_COLUMNS]


def period_comparison(data, previous_period=None, current_period=None):
    """Return the backend comparison plus an explicit availability status.

    With no selected baseline, the previous calendar month is required. Explicit
    selections may compare any two observed months in chronological order.
    Discount changes are fractions (0.05 means five percentage points).
    """
    prepared = _prepared_data(data)
    periods = available_periods(prepared)
    result = {key: None for key in _COMPARISON_VALUES}
    current_period = str(current_period) if current_period is not None else periods[-1] if periods else None
    try:
        if current_period is not None:
            current_period = str(pd.Period(current_period, freq="M"))
        if previous_period is None and current_period is not None:
            previous_period = str(pd.Period(current_period, freq="M") - 1)
        elif previous_period is not None:
            previous_period = str(pd.Period(previous_period, freq="M"))
    except (TypeError, ValueError):
        reason = "invalid_period"
    else:
        if not periods:
            reason = "no_data"
        elif current_period not in periods:
            reason = "missing_current_period"
        elif previous_period not in periods:
            reason = "missing_previous_period"
        elif previous_period >= current_period:
            reason = "invalid_period_order"
        else:
            reason = None

    result.update(
        available=reason is None,
        reason=reason,
        periodo_anterior=previous_period,
        periodo_atual=current_period,
        periodo_anterior_label=period_label(previous_period),
        periodo_atual_label=period_label(current_period),
    )
    if reason is not None:
        return result

    result.update(compare_periods(prepared, previous_period, current_period))
    for metric in ("preco_unitario_medio", "desconto_medio"):
        result[f"variacao_percentual_{metric}"] = percentage_change(
            result[f"{metric}_anterior"], result[f"{metric}_atual"]
        )
    period_values = prepared["data"].dt.to_period("M").astype(str)
    result["registros_anterior"] = int(period_values.eq(previous_period).sum())
    result["registros_atual"] = int(period_values.eq(current_period).sum())
    return result


def dimension_contributions(data, previous_period, current_period, dimension="produto", limit=8):
    """Revenue changes for one dimension, ranked by absolute contribution.

    The signed share divides each segment change by the full revenue change, not
    by the displayed top-N subtotal. Offsetting increases can therefore have a
    negative share, and a contributing decline can exceed 100%. A zero total
    change has no meaningful net-change shares. Never add rows from different
    dimensions together: products, regions and sellers overlap the same sales.
    """
    if dimension not in _DIMENSIONS:
        raise ValueError(f"Dimensão indisponível: {dimension}")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("O limite deve ser um inteiro positivo ou None.")
    summary = period_comparison(data, previous_period, current_period)
    if not summary["available"]:
        return pd.DataFrame(columns=[dimension, *_CONTRIBUTION_COLUMNS])

    prepared = _prepared_data(data)
    period_values = prepared["data"].dt.to_period("M").astype(str)

    def aggregate(period):
        grouped = (
            prepared.loc[period_values.eq(period)]
            .groupby(dimension, as_index=False, dropna=False)
            .agg(
                faturamento=("faturamento", "sum"),
                quantidade=("quantidade", "sum"),
                preco_unitario_medio=("preco_unitario", "mean"),
                desconto_medio=("desconto", "mean"),
            )
        )
        return grouped

    previous = aggregate(summary["periodo_anterior"])
    current = aggregate(summary["periodo_atual"])
    result = previous.merge(current, on=dimension, how="outer", suffixes=("_anterior", "_atual"))
    for metric in ("faturamento", "quantidade"):
        for suffix in ("anterior", "atual"):
            result[f"{metric}_{suffix}"] = result[f"{metric}_{suffix}"].fillna(0)
    for metric in ("faturamento", "quantidade", "preco_unitario_medio", "desconto_medio"):
        result[f"variacao_{metric}"] = result[f"{metric}_atual"] - result[f"{metric}_anterior"]
    result["variacao_percentual_faturamento"] = (
        result["variacao_faturamento"].div(result["faturamento_anterior"].where(result["faturamento_anterior"].ne(0))) * 100
    )
    total_change = summary["variacao_faturamento"]
    result["participacao_variacao_percentual"] = (
        result["variacao_faturamento"] / total_change * 100 if total_change != 0 else float("nan")
    )
    result["dimensao"] = dimension
    result["segmento"] = result[dimension].fillna("Não informado").astype(str)
    result["direcao"] = result["variacao_faturamento"].map(
        lambda value: "queda" if value < 0 else "aumento" if value > 0 else "estavel"
    )
    ordered_indices = result["variacao_faturamento"].abs().sort_values(ascending=False, kind="stable").index
    result = result.loc[ordered_indices]
    if limit is not None:
        result = result.head(limit)
    return result[[dimension, *_CONTRIBUTION_COLUMNS]].reset_index(drop=True)
