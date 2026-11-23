"""A single readable card for a selected historical alert."""

import json
from functools import partial

import streamlit as st

from src.analysis.alert_explanation import explain_alert
from src.agent.alert_conversation import build_alert_context
from src.data.capabilities import available_dimensions
from src.formatting import format_currency, currency_symbol
from src.evaluation.pdf_report import build_alert_pdf


def render_alert_card(data, scored, alerts, calibration, config):
    periods = alerts["periodo"].tolist()
    if st.session_state.get("historical_alert_date") not in periods:
        st.session_state.pop("historical_alert_date", None)
    period = st.selectbox("Alerta para investigar", periods, key="historical_alert_date")
    selected = scored.loc[scored["periodo"].eq(period)].iloc[0]
    facts = explain_alert(selected, calibration, config["metric"])
    facts.update(currency=data.attrs.get("dataset_context", {}).get("currency", "BRL"),
                 currency_symbol=currency_symbol(data), frequency=config["frequency"],
                 start_date=selected["inicio_periodo"].date().isoformat(),
                 end_date=selected["fim_periodo"].date().isoformat(),
                 calibration_method=calibration.get("method"))
    def value(number):
        if number is None:
            return "—"
        if config["metric"] == "faturamento":
            return format_currency(number, data)
        return f"{number:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") + " unidades"
    with st.container(border=True):
        st.markdown("**Ficha do alerta · " + period + "**")
        with st.container(horizontal=True):
            st.metric("Valor observado", value(facts["observed"]), border=True)
            st.metric("Referência esperada", value(facts["expected"]), border=True)
            st.metric("Diferença da referência", value(facts["difference"]), border=True)
        percentage = f" ({abs(facts['percentage']):.1f}%)" if facts["percentage"] is not None else ""
        st.markdown(f"O valor registrado ficou **{facts['direction']} da referência**{percentage}. A referência foi calculada com **{facts['previous_observations']} observações anteriores**.")
        if facts["score"] is not None and facts["threshold"] is not None:
            st.caption(f"O alerta foi emitido porque o escore {facts['score']:.2f} ultrapassou o limite {facts['threshold']:.2f}. Esse limite é uma regra de decisão; não é uma probabilidade de erro ou de fraude.")
        if facts["percentage"] is None:
            st.caption("A diferença percentual não é calculada para referência nula ou negativa.")
        st.markdown("**O que conferir em seguida**")
        for message in facts["next_checks"]:
            st.markdown("- " + message)
        dimensions = list(available_dimensions(data))
        observed = data.loc[data["data"].dt.normalize().between(selected["inicio_periodo"], selected["fim_periodo"])]
        if config.get("dimension") and config.get("segmento"):
            observed = observed.loc[observed[config["dimension"]].astype("string").eq(str(config["segmento"]))]
        frame = None
        if dimensions and not observed.empty:
            labels = {"produto": "Produto", "categoria": "Categoria", "regiao": data.attrs.get("dataset_context", {}).get("region_label", "Região"), "vendedor": "Vendedor"}
            if st.session_state.get("alert_detail_dimension") not in dimensions:
                st.session_state.pop("alert_detail_dimension", None)
            dimension = st.selectbox("Detalhar os registros do alerta por", dimensions, format_func=lambda key: labels[key], key="alert_detail_dimension")
            frame = observed.groupby(dimension, dropna=False)[config["metric"]].sum().nlargest(8).reset_index()
            unit = f"({currency_symbol(data)})" if config["metric"] == "faturamento" else "(unidades)"
            frame.columns = [labels[dimension], f"Valor registrado {unit}"]
            st.dataframe(frame, hide_index=True, width="stretch")
            st.caption("Esta distribuição mostra os grupos observados no período. Ela não divide a referência estatística nem confirma quem causou o desvio.")
        st.download_button("Baixar ficha do alerta", json.dumps(facts, ensure_ascii=False, indent=2),
                           file_name="ficha_alerta_" + period + ".json", mime="application/json", key="download_alert_card")
        context = build_alert_context(data, facts, config)
        context["source"].setdefault("source_label", st.session_state.get("data_label") or "Base de vendas")
        pdf_frame = frame.copy() if frame is not None else None
        if pdf_frame is not None:
            pdf_frame[pdf_frame.columns[1]] = pdf_frame.iloc[:, 1].map(value)
        st.download_button("Baixar alerta em PDF", data=partial(build_alert_pdf, context, pdf_frame),
                           file_name="alerta_" + period + ".pdf", mime="application/pdf", key="download_alert_pdf", on_click="ignore", icon=":material/picture_as_pdf:")
        if st.button("Perguntar sobre este alerta", icon=":material/forum:", type="primary", key="chat_about_alert"):
            st.session_state.alert_chat_context = context
            st.session_state.monthly_chat_context = None
            st.session_state.messages = []
            st.session_state.workspace_view = "Assistente"
            st.rerun()
    return period
