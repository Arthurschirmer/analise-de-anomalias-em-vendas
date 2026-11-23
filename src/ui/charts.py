"""Altair views of the deterministic analysis, ready for Streamlit themes."""

import math

import altair as alt
import pandas as pd

from src.formatting import currency_axis_label, currency_symbol, format_currency


TEAL = "#14B8A6"
RED = "#F06472"
AMBER = "#F5B654"
REFERENCE = "#7E8DA5"
HEIGHT = 290
MONTHS = ("", "jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _br_number(value, decimals=2):
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _money(value, context=None):
    return format_currency(value, context)


def _percent(value):
    number = _number(value)
    return "Sem comparação anterior" if number is None else f"{number:+.1f}%".replace(".", ",")


def _period_label(value):
    text = str(value or "")
    pieces = text.split("-")
    if len(pieces) == 2 and pieces[0].isdigit() and pieces[1].isdigit():
        month = int(pieces[1])
        if 1 <= month <= 12:
            return f"{MONTHS[month]}/{pieces[0]}"
    return text


def _styled(chart):
    # Text and background colors are left to st.altair_chart(theme="streamlit").
    return (
        chart.properties(height=HEIGHT)
        .configure_view(stroke=None)
        .configure_axis(domain=False, tickSize=0, gridOpacity=0.12,
                        labelFontSize=11, titleFontSize=12, labelPadding=9,
                        titlePadding=14)
        .configure_legend(title=None, labelFontSize=11, orient="bottom")
    )


def _monthly_frame(monthly):
    frame = pd.DataFrame(monthly).copy()
    defaults = {"periodo": "", "valor": None, "anomalia": False,
                "variacao_percentual": None}
    for name, default in defaults.items():
        if name not in frame:
            frame[name] = default
    frame["periodo"] = frame["periodo"].fillna("").astype(str)
    if "periodo_label" not in frame:
        frame["periodo_label"] = frame["periodo"].map(_period_label)
    frame["periodo_label"] = frame["periodo_label"].fillna(frame["periodo"])
    frame["valor"] = pd.to_numeric(frame["valor"], errors="coerce")
    frame["variacao_percentual"] = pd.to_numeric(frame["variacao_percentual"], errors="coerce")
    frame["anomalia"] = frame["anomalia"].fillna(False).astype(bool)
    if "comparavel" in frame:
        frame["comparavel"] = frame["comparavel"].fillna(False).astype(bool)
    else:
        frame["comparavel"] = frame["variacao_percentual"].notna()
    frame["comparavel"] &= frame["variacao_percentual"].notna()
    return frame.sort_values("periodo", kind="stable")


def _calendar_frame(frame):
    """Expose absent calendar months as null gaps, never as zero sales."""
    frame = frame.copy()
    if not frame.empty:
        dates = pd.to_datetime(frame["periodo"], format="%Y-%m", errors="coerce")
        if dates.notna().all() and frame["periodo"].is_unique:
            calendar = pd.period_range(dates.min(), dates.max(), freq="M").astype(str)
            frame = frame.set_index("periodo").reindex(calendar).rename_axis("periodo").reset_index()
            frame["periodo_label"] = frame["periodo_label"].fillna(frame["periodo"].map(_period_label))
            frame["anomalia"] = frame["anomalia"].fillna(False).astype(bool)
            frame["comparavel"] = frame["comparavel"].fillna(False).astype(bool)
            # A value after an absent month is not a monthly comparison, even
            # when a supplied series contains a comparison with an older row.
            frame["comparavel"] &= frame["valor"].notna() & frame["valor"].shift(1).notna()
    # Distinct path groups prevent Vega-Lite from filtering nulls and joining
    # the last point before a gap directly to the first point after it.
    frame["anomalia"] &= frame["comparavel"]
    frame["trecho"] = frame["valor"].isna().cumsum().astype(str)
    return frame


def trend_chart(monthly, metric="faturamento", context=None):
    """Show monthly values and flag only periods marked by the detector."""
    frame = _calendar_frame(_monthly_frame(monthly))
    currency_context = monthly if context is None else context
    quantity = metric == "quantidade"
    value_title = "Quantidade vendida" if quantity else "Faturamento"
    axis_title = "Unidades vendidas" if quantity else currency_axis_label(currency_context)
    frame["valor_formatado"] = frame["valor"].map(
        (lambda value: f"{_br_number(value, 0)} unidades")
        if quantity else (lambda value: _money(value, currency_context))
    )
    frame["variacao_formatada"] = [
        _percent(value) if comparable else "Sem comparação mensal"
        for value, comparable in zip(frame["variacao_percentual"], frame["comparavel"])
    ]
    frame["sinal"] = [
        ("Variação sinalizada" if flagged else "Sem sinalização")
        if comparable else "Sem comparação mensal"
        for flagged, comparable in zip(frame["anomalia"], frame["comparavel"])
    ]
    base = alt.Chart(frame).encode(
        x=alt.X("periodo_label:O", sort=frame["periodo_label"].tolist(), title=None,
                scale=alt.Scale(domain=frame["periodo_label"].tolist()),
                axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("valor:Q", title=axis_title, scale=alt.Scale(zero=True),
                axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("periodo_label:N", title="Mês"),
                 alt.Tooltip("valor_formatado:N", title=value_title),
                 alt.Tooltip("variacao_formatada:N", title="Variação mensal"),
                 alt.Tooltip("sinal:N", title="Detecção")],
    )
    area = base.mark_area(color=TEAL, opacity=0.09).encode(detail="trecho:N")
    line = base.mark_line(color=TEAL, strokeWidth=3).encode(detail="trecho:N")
    points = base.mark_point(filled=True).encode(
        color=alt.Color("sinal:N", title=None,
                        scale=alt.Scale(domain=["Sem comparação mensal", "Sem sinalização", "Variação sinalizada"],
                                        range=[REFERENCE, TEAL, AMBER])),
        size=alt.condition(alt.datum.anomalia, alt.value(150), alt.value(65)),
    )
    return _styled(area + line + points)


def change_chart(monthly):
    """Show monthly percentage changes, preserving undefined first periods."""
    frame = _calendar_frame(_monthly_frame(monthly))
    frame = frame[frame["comparavel"] & frame["variacao_percentual"].notna()].copy()
    frame["variacao_formatada"] = frame["variacao_percentual"].map(_percent)
    bars = alt.Chart(frame).mark_bar(cornerRadius=4).encode(
        x=alt.X("periodo_label:O", title=None, sort=frame["periodo_label"].tolist(),
                axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("variacao_percentual:Q", title="Variação mensal (%)",
                axis=alt.Axis(labelExpr="format(datum.value, '+.0f') + '%'")),
        color=alt.condition(alt.datum.variacao_percentual < 0, alt.value(RED), alt.value(TEAL)),
        tooltip=[alt.Tooltip("periodo_label:N", title="Mês"),
                 alt.Tooltip("variacao_formatada:N", title="Variação")],
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
        color=REFERENCE, strokeOpacity=0.5
    ).encode(y="zero:Q")
    return _styled(bars + zero)


def contribution_chart(frame, previous_label, current_label, context=None):
    """Compare revenue changes within one selected dimension.

    The caller supplies segments from a single dimension, avoiding sums across
    overlapping products, categories, sellers and regions.
    """
    data = pd.DataFrame(frame).copy()
    currency_context = frame if context is None else context
    for name in ("segmento", "variacao_faturamento", "faturamento_anterior", "faturamento_atual"):
        if name not in data:
            data[name] = pd.Series(dtype="str" if name == "segmento" else "float64")
    data["segmento"] = data["segmento"].fillna("Sem identificação").astype(str)
    for name in ("variacao_faturamento", "faturamento_anterior", "faturamento_atual"):
        data[name] = pd.to_numeric(data[name], errors="coerce")
        data[f"{name}_formatado"] = data[name].map(lambda value: _money(value, currency_context))
    bars = alt.Chart(data).mark_bar(cornerRadius=3).encode(
        x=alt.X("variacao_faturamento:Q", title=f"Variação do faturamento ({currency_symbol(currency_context)})",
                scale=alt.Scale(zero=True), axis=alt.Axis(format="~s")),
        y=alt.Y("segmento:N", title=None, sort=data["segmento"].tolist(),
                axis=alt.Axis(grid=False, labelLimit=160)),
        color=alt.condition(alt.datum.variacao_faturamento < 0, alt.value(RED), alt.value(TEAL)),
        tooltip=[alt.Tooltip("segmento:N", title="Segmento"),
                 alt.Tooltip("faturamento_anterior_formatado:N", title=f"Antes · {previous_label}"),
                 alt.Tooltip("faturamento_atual_formatado:N", title=f"Depois · {current_label}"),
                 alt.Tooltip("variacao_faturamento_formatado:N", title="Variação")],
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(
        color=REFERENCE, strokeOpacity=0.5
    ).encode(x="zero:Q")
    return _styled(bars + zero)


def comparison_chart(comparison, context=None):
    """Show previous/current revenue from a report or its comparison section."""
    comparison = comparison or {}
    currency_context = comparison if context is None else context
    summary = comparison.get("resumo_comparativo", comparison)
    periods = comparison.get("periodos", comparison)
    previous = summary.get("faturamento_anterior", summary.get("valor_anterior"))
    current = summary.get("faturamento_atual", summary.get("valor_atual"))
    previous_label = periods.get("periodo_anterior") or summary.get("periodo_anterior") or "Referência"
    current_label = periods.get("periodo_atual") or summary.get("periodo_atual") or "Em análise"
    data = pd.DataFrame([
        {"papel": "Referência", "periodo": _period_label(previous_label), "valor": _number(previous)},
        {"papel": "Em análise", "periodo": _period_label(current_label), "valor": _number(current)},
    ])
    data["valor_formatado"] = data["valor"].map(lambda value: _money(value, currency_context))
    base = alt.Chart(data).encode(
        x=alt.X("papel:N", sort=["Referência", "Em análise"], title=None,
                axis=alt.Axis(labelAngle=0, grid=False)),
        y=alt.Y("valor:Q", title=currency_axis_label(currency_context), scale=alt.Scale(zero=True),
                axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("periodo:N", title="Período"),
                 alt.Tooltip("valor_formatado:N", title="Faturamento")],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        color=alt.Color("papel:N", scale=alt.Scale(domain=["Referência", "Em análise"],
                                                  range=[REFERENCE, TEAL]), legend=None)
    )
    labels = base.mark_text(dy=-7, baseline="bottom", fontSize=12, fontWeight=600).encode(
        text="valor_formatado:N"
    )
    return _styled(bars + labels)
