"""Exact accounting of a revenue change across two observed months.

The decomposition is descriptive. It separates the matched product basket from
entries/exits, and never invents a comparison price for an absent product.
"""

import math

import pandas as pd


_COMPONENTS = (
    ("volume", "Volume de itens na cesta comum"),
    ("mix", "Composição de produtos na cesta comum"),
    ("unit_value", "Valor unitário efetivo na cesta comum"),
    ("entries", "Produtos observados só no período atual"),
    ("exits", "Produtos observados só na referência"),
)
_CONVENTION = (
    "Decomposição sequencial: volume ao mix e valor unitário da referência; "
    "depois mix ao volume atual e valor da referência; depois valor unitário "
    "nas quantidades atuais. Produtos presentes em apenas um mês são separados."
)
_LIMITATIONS = [
    "A decomposição reconcilia uma identidade contábil; não identifica causalidade.",
    "Os efeitos dependem da ordem declarada de volume, mix e valor unitário.",
    "Valor unitário efetivo é faturamento dividido pela quantidade do mesmo produto; não é necessariamente preço de tabela.",
    "Descontos, frete e definição de faturamento dependem da fonte. Campos de preenchimento não são observações.",
    "Entrada e saída significam presença nos dois meses escolhidos, não lançamento ou retirada definitiva de um produto.",
]


def _unavailable(reason, previous=None, current=None, detail=None):
    return {
        "available": False,
        "reason": reason,
        "periods": {"periodo_anterior": previous, "periodo_atual": current},
        "totals": {},
        "components": [],
        "component_values": {},
        "product_changes": [],
        "product_changes_total": 0,
        "product_changes_truncated": False,
        "reconciliation_error": None,
        "reconciled": False,
        "convention": _CONVENTION,
        "limitations": [*_LIMITATIONS, *([detail] if detail else [])],
    }


def _month(value):
    period = pd.Period(value, freq="M")
    if str(period) != str(value):
        raise ValueError("Use meses no formato AAAA-MM.")
    return str(period)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def decompose_revenue_change(data, previous_period, current_period, *, tolerance=1e-6, max_products=50):
    """Separate volume, product mix, effective unit value and observed entries/exits.

    Quantities/revenues must be finite and non-negative. A month with zero total
    quantity has no meaningful unit value, so the result is unavailable. Revenue
    may be zero with positive quantities; that does not require division by zero.
    Returned product rows are the largest absolute changes, while all effects
    and reconciliation use the complete product set.
    """
    if not _finite(tolerance) or tolerance < 0:
        raise ValueError("A tolerância deve ser finita e não negativa.")
    if max_products is not None and (
        not isinstance(max_products, int) or isinstance(max_products, bool) or max_products <= 0
    ):
        raise ValueError("O limite de produtos deve ser um inteiro positivo ou None.")
    try:
        previous, current = _month(previous_period), _month(current_period)
    except (TypeError, ValueError):
        return _unavailable("invalid_period", str(previous_period), str(current_period))
    if previous >= current:
        return _unavailable("invalid_period_order", previous, current)
    required = {"data", "produto", "quantidade", "faturamento"}
    if data is None or not required.issubset(data.columns):
        return _unavailable("missing_columns", previous, current)
    if data.empty:
        return _unavailable("no_data", previous, current)

    frame = data[list(required)].copy()
    frame["data"] = pd.to_datetime(frame["data"], errors="coerce")
    if frame["data"].isna().any():
        return _unavailable("invalid_dates", previous, current)
    month = frame["data"].dt.to_period("M").astype(str)
    frame = frame.loc[month.isin([previous, current])].copy()
    frame["periodo"] = frame["data"].dt.to_period("M").astype(str)
    if not frame["periodo"].eq(previous).any():
        return _unavailable("missing_previous_period", previous, current)
    if not frame["periodo"].eq(current).any():
        return _unavailable("missing_current_period", previous, current)
    if frame["produto"].isna().any():
        return _unavailable("missing_product_identifiers", previous, current)
    for column in ("quantidade", "faturamento"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame[column].map(_finite).all():
            return _unavailable("invalid_numeric_values", previous, current)
        if frame[column].lt(0).any():
            return _unavailable(
                "negative_sales_not_supported", previous, current,
                "Quantidades ou receitas negativas exigem uma convenção específica para devoluções.",
            )
    if (frame["quantidade"].eq(0) & frame["faturamento"].ne(0)).any():
        return _unavailable("revenue_without_quantity", previous, current)
    frame["produto"] = frame["produto"].astype(str)
    grouped = (
        frame.groupby(["periodo", "produto"], as_index=False)
        .agg(quantidade=("quantidade", "sum"), faturamento=("faturamento", "sum"))
    )
    # Zero-quantity/zero-revenue rows have no unit value or observed contribution.
    active = grouped.loc[grouped["quantidade"].gt(0)].copy()
    baskets = {}
    for period in (previous, current):
        part = active.loc[active["periodo"].eq(period)].set_index("produto")
        if part.empty or part["quantidade"].sum() <= 0:
            return _unavailable("zero_total_quantity", previous, current)
        if not all(_finite(value) for value in (part["quantidade"].sum(), part["faturamento"].sum())):
            return _unavailable("nonfinite_totals", previous, current)
        part["unit_value"] = part["faturamento"] / part["quantidade"]
        baskets[period] = part

    old, new = baskets[previous], baskets[current]
    common = sorted(set(old.index) & set(new.index))
    entries = sorted(set(new.index) - set(old.index))
    exits = sorted(set(old.index) - set(new.index))
    q0, q1 = float(old["quantidade"].sum()), float(new["quantidade"].sum())
    r0, r1 = float(old["faturamento"].sum()), float(new["faturamento"].sum())
    common_q0 = float(old.loc[common, "quantidade"].sum()) if common else 0.0
    common_q1 = float(new.loc[common, "quantidade"].sum()) if common else 0.0
    values = dict.fromkeys((key for key, _ in _COMPONENTS), 0.0)
    changes = []
    for product in common:
        old_row, new_row = old.loc[product], new.loc[product]
        qi0, qi1 = float(old_row["quantidade"]), float(new_row["quantidade"])
        pi0, pi1 = float(old_row["unit_value"]), float(new_row["unit_value"])
        share0, share1 = qi0 / common_q0, qi1 / common_q1
        volume = (common_q1 - common_q0) * share0 * pi0
        mix = common_q1 * (share1 - share0) * pi0
        unit_value = qi1 * (pi1 - pi0)
        values["volume"] += volume
        values["mix"] += mix
        values["unit_value"] += unit_value
        changes.append({
            "produto": product, "status": "comum",
            "quantidade_anterior": qi0, "quantidade_atual": qi1,
            "valor_unitario_anterior": pi0, "valor_unitario_atual": pi1,
            "faturamento_anterior": float(old_row["faturamento"]),
            "faturamento_atual": float(new_row["faturamento"]),
            "variacao_faturamento": float(new_row["faturamento"] - old_row["faturamento"]),
            "efeito_volume": volume, "efeito_mix": mix, "efeito_valor_unitario": unit_value,
            "efeito_entrada": 0.0, "efeito_saida": 0.0,
        })
    for status, products, basket in (
        ("entrada_observada", entries, new), ("saida_observada", exits, old),
    ):
        for product in products:
            row = basket.loc[product]
            is_entry = status == "entrada_observada"
            revenue, quantity, price = float(row["faturamento"]), float(row["quantidade"]), float(row["unit_value"])
            effect = revenue if is_entry else -revenue
            values["entries" if is_entry else "exits"] += effect
            changes.append({
                "produto": product, "status": status,
                "quantidade_anterior": 0.0 if is_entry else quantity,
                "quantidade_atual": quantity if is_entry else 0.0,
                "valor_unitario_anterior": None if is_entry else price,
                "valor_unitario_atual": price if is_entry else None,
                "faturamento_anterior": 0.0 if is_entry else revenue,
                "faturamento_atual": revenue if is_entry else 0.0,
                "variacao_faturamento": effect,
                "efeito_volume": 0.0, "efeito_mix": 0.0, "efeito_valor_unitario": 0.0,
                "efeito_entrada": effect if is_entry else 0.0,
                "efeito_saida": 0.0 if is_entry else effect,
            })

    total_change = r1 - r0
    accounted = math.fsum(values.values())
    error = accounted - total_change
    reconciled = math.isclose(accounted, total_change, abs_tol=tolerance, rel_tol=1e-9)
    if not all(_finite(value) for value in [*values.values(), accounted, error]):
        return _unavailable("nonfinite_totals", previous, current)
    for change in changes:
        change["participacao_variacao_percentual"] = (
            change["variacao_faturamento"] / total_change * 100 if total_change != 0 else None
        )
    changes.sort(key=lambda item: (-abs(item["variacao_faturamento"]), item["produto"]))
    all_changes_count = len(changes)
    if max_products is not None:
        changes = changes[:max_products]
    limitations = list(_LIMITATIONS)
    if not common:
        limitations.append(
            "Não há produtos comuns nos dois meses. Volume, mix e valor unitário da cesta comum não são estimáveis; a diferença é explicada contabilmente por entradas e saídas observadas."
        )
    if data.attrs.get("dataset_context", {}).get("discount_available") is False:
        limitations.append("Descontos não estão disponíveis nesta fonte; nenhum efeito foi atribuído a descontos.")
    return {
        "available": reconciled,
        "reason": None if reconciled else "reconciliation_failed",
        "periods": {"periodo_anterior": previous, "periodo_atual": current},
        "totals": {
            "faturamento_anterior": r0, "faturamento_atual": r1,
            "variacao_faturamento": total_change,
            "quantidade_anterior": q0, "quantidade_atual": q1,
            "valor_unitario_efetivo_anterior": r0 / q0,
            "valor_unitario_efetivo_atual": r1 / q1,
            "quantidade_cesta_comum_anterior": common_q0,
            "quantidade_cesta_comum_atual": common_q1,
            "produtos_comuns": len(common), "entradas_observadas": len(entries),
            "saidas_observadas": len(exits), "soma_efeitos": accounted,
        },
        "common_basket_available": bool(common),
        "components": [
            {"key": key, "label": label, "value": float(values[key]),
             "available": bool(common) if key in {"volume", "mix", "unit_value"} else True}
            for key, label in _COMPONENTS
        ],
        "component_values": {key: float(value) for key, value in values.items()},
        "product_changes": changes,
        "product_changes_total": all_changes_count,
        "product_changes_truncated": all_changes_count > len(changes),
        "reconciliation_error": error,
        "reconciled": reconciled,
        "convention": _CONVENTION,
        "limitations": limitations,
    }

