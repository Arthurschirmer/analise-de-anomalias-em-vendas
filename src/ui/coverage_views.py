"""Keep observational coverage next to the comparison it qualifies."""

import streamlit as st

from src.analysis.period_coverage import comparison_coverage


def render_comparison_coverage(data, previous, current):
    try:
        coverage = comparison_coverage(data, previous, current)
    except ValueError:
        return None
    if coverage["needs_attention"]:
        st.warning("As datas observadas não cobrem as bordas de um ou dos dois meses. A diferença pode refletir o recorte. Isso não comprova falha de coleta nem ausência de vendas.")
    for part in (coverage["previous"], coverage["current"]):
        st.caption(f"{part['period']}: {part['first_date']} a {part['last_date']} · {part['observed_days']} dias com registros · {part['calendar_days']} dias no calendário.")
    return coverage
