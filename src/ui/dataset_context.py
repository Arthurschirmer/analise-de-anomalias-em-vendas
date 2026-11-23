"""Dataset meaning and observational coverage for the interface and model prompts.

Coverage is inferred from observed rows, not from a collection contract. Sparse
months and boundary dates must not be presented as proof that data is incomplete.
"""

import math

import pandas as pd

from src.data.capabilities import available_dimensions
from src.formatting import currency_symbol


SPARSE_MONTH_QUANTITY = 100
_OLIST_REVENUE = "Soma do preço dos itens de pedidos entregues, pela data da compra; frete excluído."
_OLIST_QUANTITY = "Soma de itens de pedidos entregues; um pedido pode conter vários itens."
_OLIST_ROWS = (
    "Cada linha padronizada pode agrupar itens com a mesma data, produto, categoria, "
    "UF do cliente e vendedor; não representa necessariamente um pedido."
)


def _finite_sum(data, column):
    if data.empty:
        return 0.0
    if column not in data:
        return None
    value = pd.to_numeric(data[column], errors="coerce").sum(min_count=1)
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def _metadata(data):
    attrs = dict(getattr(data, "attrs", {}) or {})
    nested = attrs.get("dataset_context", {})
    if isinstance(nested, dict):
        attrs.update(nested)
    return attrs


def _category_coverage(data):
    if "categoria" not in data or _metadata(data).get("category_available") is False:
        missing = data
    else:
        categories = data["categoria"].astype("string").str.strip().str.casefold()
        mask = categories.isna() | categories.isin(["", "sem_categoria", "sem categoria"])
        missing = data.loc[mask]
    return {
        "row_count": int(len(missing)),
        "units": _finite_sum(missing, "quantidade"),
        "revenue": _finite_sum(missing, "faturamento"),
    }


def _monthly_coverage(data):
    result = {
        "first_date": None,
        "last_date": None,
        "observed_months": 0,
        "observed_periods": [],
        "missing_months": [],
        "sparse_months": [],
        "sparse_threshold_quantity": SPARSE_MONTH_QUANTITY,
        "boundary_months": [],
        "possible_partial_months": [],
    }
    if data.empty or "data" not in data:
        return result
    dates = pd.to_datetime(data["data"], errors="coerce")
    valid = dates.notna()
    if not valid.any():
        return result

    observed_dates = dates.loc[valid]
    months = observed_dates.dt.to_period("M").astype(str)
    observed_periods = sorted(months.unique().tolist())
    first_date, last_date = observed_dates.min(), observed_dates.max()
    result.update(
        first_date=first_date.date().isoformat(),
        last_date=last_date.date().isoformat(),
        observed_months=len(observed_periods),
        observed_periods=observed_periods,
    )
    calendar_months = pd.period_range(observed_periods[0], observed_periods[-1], freq="M").astype(str)
    result["missing_months"] = [month for month in calendar_months if month not in observed_periods]

    if "quantidade" in data:
        monthly = pd.DataFrame({
            "periodo": months,
            "quantidade": pd.to_numeric(data.loc[valid, "quantidade"], errors="coerce"),
        }).groupby("periodo", as_index=False).agg(
            quantidade=("quantidade", lambda values: values.sum(min_count=1)),
            registros=("periodo", "size"),
        )
        result["sparse_months"] = [
            {"periodo": str(row.periodo), "quantidade": float(row.quantidade), "registros": int(row.registros)}
            for row in monthly.itertuples(index=False)
            if pd.notna(row.quantidade) and math.isfinite(float(row.quantidade))
            and row.quantidade < SPARSE_MONTH_QUANTITY
        ]

    boundary_periods = list(dict.fromkeys([observed_periods[0], observed_periods[-1]]))
    for period in boundary_periods:
        period_dates = observed_dates.loc[months.eq(period)]
        first_observed, last_observed = period_dates.min(), period_dates.max()
        starts_late = first_observed.day > 1
        ends_early = last_observed.day < last_observed.days_in_month
        is_first = period == observed_periods[0]
        is_last = period == observed_periods[-1]
        possible_partial = (is_first and starts_late) or (is_last and ends_early)
        result["boundary_months"].append({
            "periodo": period,
            "first_observed_date": first_observed.date().isoformat(),
            "last_observed_date": last_observed.date().isoformat(),
            "starts_after_month_start": bool(starts_late),
            "ends_before_month_end": bool(ends_early),
            "is_first_month": bool(is_first),
            "is_last_month": bool(is_last),
            "possibly_partial": bool(possible_partial),
        })
        if possible_partial:
            result["possible_partial_months"].append(period)
    return result


def dataset_context(data, source_type):
    """Return serializable meaning, coverage notes and compact AI constraints.

    The standard schema alone does not establish the business definition of a
    custom CSV. In particular, zero discounts do not imply unavailable discount
    data. The Olist adapter, however, explicitly creates the zero placeholder.
    Adapter metadata is read from data.attrs['dataset_context'] when present.
    """
    data = data if data is not None else pd.DataFrame()
    metadata = _metadata(data)
    source = str(source_type or metadata.get("source_type") or "csv").strip().casefold()
    is_olist = source == "olist"
    source_label = (
        "Base pública · Olist" if is_olist
        else "Dados fictícios" if source == "demo"
        else str(metadata.get("source_label") or metadata.get("label") or "Base própria")
    )
    discount_attribute = metadata.get("discount_available")
    discount_available = (
        False if is_olist or discount_attribute is False
        else True if "desconto" in data
        else None
    )
    statuses = metadata.get("order_statuses")
    custom_statuses = is_olist and statuses is not None and set(statuses) != {"delivered"}
    olist_revenue = (
        "Soma do preço dos itens dos pedidos com os status selecionados, pela data da compra; frete excluído."
        if custom_statuses else _OLIST_REVENUE
    )
    olist_quantity = (
        "Soma de itens dos pedidos com os status selecionados; um pedido pode conter vários itens."
        if custom_statuses else _OLIST_QUANTITY
    )
    revenue_definition = str(metadata.get("revenue_definition") or (
        olist_revenue if is_olist
        else "Soma da coluna faturamento da base enviada; a definição comercial depende do arquivo de origem."
    ))
    quantity_definition = str(metadata.get("quantity_definition") or (
        olist_quantity if is_olist
        else "Soma da coluna quantidade; sua unidade depende da base de origem."
    ))
    row_definition = str(metadata.get("row_definition") or (
        _OLIST_ROWS if is_olist
        else "Um registro do arquivo padronizado; a granularidade depende da base de origem."
    ))
    region_label = "UF do cliente" if is_olist else str(metadata.get("region_label") or "Região")
    product_label = "ID do produto" if is_olist else str(metadata.get("product_label") or "Produto")
    seller_label = "ID do vendedor" if is_olist else str(metadata.get("seller_label") or "Vendedor")
    dimensions = list(available_dimensions(metadata))
    currency = str(metadata.get("currency") or "BRL")
    coverage = _monthly_coverage(data)
    missing_category = _category_coverage(data)
    notes = []
    restrictions = [
        "As evidências mostram associações. Não confirme causalidade nem precisão da detecção sem avaliação rotulada.",
        "Meses sem registros não são automaticamente meses com vendas zero.",
        "Baixo volume e datas de borda são alertas de cobertura observada, não prova de coleta incompleta.",
        "Não calcule lucro ou custo sem as respectivas colunas e definições.",
    ]
    if is_olist:
        notes.extend([
            revenue_definition,
            quantity_definition,
            row_definition,
            "A dimensão Região contém a UF do cliente; produtos e vendedores são identificadores, não nomes.",
            "Desconto não foi observado na Olist: o valor 0 da adaptação é um preenchimento técnico.",
        ])
        restrictions.extend([
            "A dimensão regiao contém UFs do cliente; não a interprete como macrorregião brasileira.",
            "Produto e vendedor são IDs; não invente nomes de produtos ou pessoas.",
            "Desconto é indisponível: o zero padronizado é um preenchimento técnico, não desconto medido. Não atribua variações a descontos.",
            "Faturamento é valor dos itens dos pedidos selecionados pela data da compra; exclui frete e não significa lucro.",
        ])
    elif discount_available is False:
        notes.append("A origem da base declara desconto indisponível; não interprete valores de preenchimento como descontos observados.")
        restrictions.append("A origem declara desconto indisponível. Não atribua variações a valores de preenchimento dessa coluna.")

    if source == "online_retail":
        notes.extend([revenue_definition, quantity_definition, row_definition])
        notes.extend(str(item) for item in metadata.get("limitations", []))
    if currency != "BRL":
        money_note = f"Valores monetários em {currency} ({currency_symbol(metadata)}), sem conversão para reais."
        notes.append(money_note)
        restrictions.append(money_note)
    for dimension, label in (("categoria", "Categoria"), ("vendedor", "Vendedor")):
        if dimension not in dimensions:
            note = f"{label} não é informado pela fonte. O marcador na coluna é técnico e não permite análises por essa dimensão."
            notes.append(note)
            restrictions.append(note)

    if coverage["missing_months"]:
        notes.append("Meses sem registros no intervalo observado: " + ", ".join(coverage["missing_months"]) + ". Eles não foram preenchidos com zero.")
    if coverage["sparse_months"]:
        periods = ", ".join(item["periodo"] for item in coverage["sparse_months"])
        notes.append(
            f"Meses com menos de {SPARSE_MONTH_QUANTITY} unidades: {periods}. "
            "Esse limiar é exploratório e não demonstra que a coleta está incompleta."
        )
    if coverage["possible_partial_months"]:
        periods = ", ".join(coverage["possible_partial_months"])
        notes.append(
            f"As datas observadas não cobrem as bordas do calendário em {periods}. "
            "Isso pode refletir o recorte de dados ou dias sem venda; não confirma ausência de registros."
        )
    if missing_category["row_count"]:
        notes.append(
            f"{missing_category['row_count']} registros padronizados estão sem categoria informada. "
            "sem_categoria é um marcador de ausência, não uma categoria comercial."
        )
        restrictions.append("sem_categoria indica informação ausente; não a trate como uma categoria comercial conhecida.")

    ai_context = {
        "fonte": source_label,
        "faturamento": revenue_definition,
        "quantidade": quantity_definition,
        "granularidade": row_definition,
        "dimensoes": {"regiao": region_label, "produto": product_label, "vendedor": seller_label},
        "desconto_disponivel": discount_available,
        "moeda": currency,
        "dimensoes_disponiveis": dimensions,
        "cobertura": {
            "primeira_data": coverage["first_date"],
            "ultima_data": coverage["last_date"],
            "meses_observados": coverage["observed_months"],
            "meses_ausentes": coverage["missing_months"],
            "meses_baixo_volume": [item["periodo"] for item in coverage["sparse_months"]],
            "limiar_exploratorio_unidades": SPARSE_MONTH_QUANTITY,
            "meses_borda_possivelmente_parciais": coverage["possible_partial_months"],
        },
        "categoria_ausente": missing_category,
        "restricoes": restrictions,
    }
    if metadata.get("order_statuses") is not None:
        ai_context["status_pedidos_incluidos"] = list(metadata["order_statuses"])
    return {
        "source_label": source_label,
        "source_type": source,
        "currency": currency,
        "currency_symbol": currency_symbol(metadata),
        "available_dimensions": dimensions,
        "category_available": "categoria" in dimensions,
        "seller_available": "vendedor" in dimensions,
        "region_label": region_label,
        "product_label": product_label,
        "seller_label": seller_label,
        "discount_available": discount_available,
        "revenue_definition": revenue_definition,
        "quantity_definition": quantity_definition,
        "row_definition": row_definition,
        "monthly_coverage": coverage,
        "missing_category": missing_category,
        "notes": notes,
        "ai_context": ai_context,
    }

