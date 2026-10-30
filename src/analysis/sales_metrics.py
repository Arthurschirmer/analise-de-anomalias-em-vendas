import pandas as pd


def total_revenue(data):
    return float(data["faturamento"].sum())


def total_quantity(data):
    return int(data["quantidade"].sum())


def sales_by_period(data, freq="M"):
    grouped = (
        data.assign(periodo=data["data"].dt.to_period(freq).astype(str))
        .groupby("periodo", as_index=False)
        .agg(
            faturamento=("faturamento", "sum"),
            quantidade=("quantidade", "sum"),
            preco_unitario_medio=("preco_unitario", "mean"),
            desconto_medio=("desconto", "mean"),
        )
        .sort_values("periodo")
    )
    return grouped


def sales_by_product(data):
    return _sales_by_dimension(data, "produto")


def sales_by_category(data):
    return _sales_by_dimension(data, "categoria")


def sales_by_region(data):
    return _sales_by_dimension(data, "regiao")


def sales_by_seller(data):
    return _sales_by_dimension(data, "vendedor")


def percentage_change(previous, current):
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100


def filter_by_period(data, period):
    period_values = data["data"].dt.to_period("M").astype(str)
    return data.loc[period_values == period].copy()


def compare_periods(data, previous_period, current_period):
    previous_data = filter_by_period(data, previous_period)
    current_data = filter_by_period(data, current_period)

    previous_revenue = total_revenue(previous_data)
    current_revenue = total_revenue(current_data)
    previous_quantity = total_quantity(previous_data)
    current_quantity = total_quantity(current_data)
    previous_discount = float(previous_data["desconto"].mean()) if not previous_data.empty else 0.0
    current_discount = float(current_data["desconto"].mean()) if not current_data.empty else 0.0
    previous_price = (
        float(previous_data["preco_unitario"].mean()) if not previous_data.empty else 0.0
    )
    current_price = (
        float(current_data["preco_unitario"].mean()) if not current_data.empty else 0.0
    )

    return {
        "periodo_anterior": previous_period,
        "periodo_atual": current_period,
        "faturamento_anterior": previous_revenue,
        "faturamento_atual": current_revenue,
        "variacao_faturamento": current_revenue - previous_revenue,
        "variacao_percentual_faturamento": percentage_change(
            previous_revenue, current_revenue
        ),
        "quantidade_anterior": previous_quantity,
        "quantidade_atual": current_quantity,
        "variacao_quantidade": current_quantity - previous_quantity,
        "variacao_percentual_quantidade": percentage_change(
            previous_quantity, current_quantity
        ),
        "desconto_medio_anterior": previous_discount,
        "desconto_medio_atual": current_discount,
        "variacao_desconto_medio": current_discount - previous_discount,
        "preco_unitario_medio_anterior": previous_price,
        "preco_unitario_medio_atual": current_price,
        "variacao_preco_unitario_medio": current_price - previous_price,
    }


def compare_periods_by_dimension(data, previous_period, current_period, dimension):
    previous_data = filter_by_period(data, previous_period)
    current_data = filter_by_period(data, current_period)

    previous_grouped = _sales_by_dimension(previous_data, dimension).rename(
        columns={
            "faturamento": "faturamento_anterior",
            "quantidade": "quantidade_anterior",
            "preco_unitario_medio": "preco_unitario_medio_anterior",
            "desconto_medio": "desconto_medio_anterior",
        }
    )
    current_grouped = _sales_by_dimension(current_data, dimension).rename(
        columns={
            "faturamento": "faturamento_atual",
            "quantidade": "quantidade_atual",
            "preco_unitario_medio": "preco_unitario_medio_atual",
            "desconto_medio": "desconto_medio_atual",
        }
    )

    comparison = pd.merge(
        previous_grouped,
        current_grouped,
        on=dimension,
        how="outer",
    ).fillna(0)

    comparison["variacao_faturamento"] = (
        comparison["faturamento_atual"] - comparison["faturamento_anterior"]
    )
    comparison["variacao_percentual_faturamento"] = comparison.apply(
        lambda row: percentage_change(
            row["faturamento_anterior"], row["faturamento_atual"]
        ),
        axis=1,
    )
    comparison["variacao_quantidade"] = (
        comparison["quantidade_atual"] - comparison["quantidade_anterior"]
    )
    comparison["variacao_desconto_medio"] = (
        comparison["desconto_medio_atual"] - comparison["desconto_medio_anterior"]
    )
    comparison["variacao_preco_unitario_medio"] = (
        comparison["preco_unitario_medio_atual"]
        - comparison["preco_unitario_medio_anterior"]
    )

    return comparison.sort_values("variacao_faturamento")


def _sales_by_dimension(data, dimension):
    return (
        data.groupby(dimension, as_index=False)
        .agg(
            faturamento=("faturamento", "sum"),
            quantidade=("quantidade", "sum"),
            preco_unitario_medio=("preco_unitario", "mean"),
            desconto_medio=("desconto", "mean"),
        )
        .sort_values("faturamento", ascending=False)
    )
