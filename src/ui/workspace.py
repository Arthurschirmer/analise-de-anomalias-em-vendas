"""Streamlit workspace for inspecting and investigating sales anomalies."""

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.conversational_analytics import answer_sales_question, build_entity_index, normalize_text
from src.agent.investigation_agent import InvestigationAgent
from src.agent.period_parser import extract_periods_from_text
from src.analysis.anomaly_detection import build_anomaly_description, detect_period_anomalies, get_most_relevant_anomaly
from src.analysis.sales_metrics import sales_by_period, total_quantity, total_revenue
from src.data.adapters import load_sales_dataset
from src.data.loader import prepare_sales_data
from src.llm.factory import create_llm_provider_from_env
from src.llm.mock_provider import MockLLMProvider
from src.ui.analytics import available_periods, dimension_contributions, monthly_overview, period_comparison, period_label
from src.ui.charts import change_chart, comparison_chart, contribution_chart, trend_chart
from src.ui.dataset_context import dataset_context
from src.ui.feedback import classify_error, sanitize_details
from src.ui.research_views import render_history, render_evaluation, render_revenue_explanation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCENARIOS = {
    "Queda nas vendas": ("data/vendas_demo.csv", "Julho e agosto de 2024 · 16 registros fictícios"),
    "Variação regional": ("data/vendas_queda_regiao_sul.csv", "Agosto e setembro de 2024 · cenário regional"),
    "Mudança nos descontos": ("data/vendas_aumento_desconto.csv", "Setembro e outubro de 2024 · cenário comercial"),
    "Mudança nos preços": ("data/vendas_queda_preco_monitor.csv", "Outubro e novembro de 2024 · cenário de preços"),
}
DIMENSIONS = {"Produto": "produto", "Categoria": "categoria", "Região": "regiao", "Vendedor": "vendedor"}
COLUMN_LABELS = {
    "data": "Data", "produto": "Produto", "categoria": "Categoria", "regiao": "Região",
    "vendedor": "Vendedor", "quantidade": "Quantidade", "preco_unitario": "Preço unitário",
    "desconto": "Desconto", "faturamento": "Faturamento", "periodo": "Período",
    "faturamento_anterior": "Faturamento anterior", "faturamento_atual": "Faturamento atual",
    "variacao_faturamento": "Variação em R$", "variacao_percentual_faturamento": "Variação (%)",
}
NAVIGATION = {"Painel": "dashboard", "Histórico": "timeline", "Investigação": "troubleshoot", "Assistente": "forum", "Avaliação": "science", "Dados e relatório": "description"}


def main():
    _init_state()
    _render_sidebar()
    _render_header()
    data = st.session_state.data
    if data is None or data.empty:
        _render_empty_state()
        return
    monthly = monthly_overview(data, threshold_pct=st.session_state.threshold_pct)
    _render_dataset_context(data)
    _render_data_overview(data, monthly)
    view = st.segmented_control(
        "Área de trabalho", list(NAVIGATION), default="Painel" if "workspace_view" not in st.session_state else None, required=True,
        format_func=lambda label: f":material/{NAVIGATION[label]}: {label}",
        key="workspace_view", label_visibility="collapsed", width="stretch",
    )
    if view == "Investigação":
        _render_investigation(data)
    elif view == "Histórico":
        render_history(data, lambda demo: _set_dataset(demo, "controlled_history", "Histórico fictício · 420 dias", "demo"))
    elif view == "Avaliação":
        render_evaluation(data, st.session_state.last_investigation, create_llm_provider_from_env)
    elif view == "Assistente":
        _render_chat(data)
    elif view == "Dados e relatório":
        _render_data_and_report(data)
    else:
        _render_dashboard(data, monthly)


@st.cache_data(ttl=600, max_entries=6, show_spinner=False)
def _read_local_dataset(dataset_type, csv_path=None, input_dir=None):
    return load_sales_dataset(dataset_type=dataset_type, csv_path=csv_path, input_dir=input_dir)


def _init_state():
    defaults = {
        "data": None, "data_source": None, "data_label": None, "data_source_type": None,
        "messages": [], "last_investigation": None, "entity_index": None, "available_periods": None,
        "threshold_pct": 30.0, "last_investigation_feedback": None, "last_investigation_mode": "local",
        "last_api_feedback": None, "ai_status": "configured", "load_feedback": None,
        "detection_results": None, "detection_feedback": None, "investigation_time": None,
        "startup_attempted": False, "data_context": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "threshold_input" in st.session_state:
        st.session_state.threshold_pct = float(st.session_state.threshold_input)
    if not st.session_state.startup_attempted:
        st.session_state.startup_attempted = True
        if st.session_state.data is None:
            source, _ = DEMO_SCENARIOS["Queda nas vendas"]
            _load_dataset(
                lambda: _read_local_dataset("csv", csv_path=str(PROJECT_ROOT / source)),
                source, "Queda nas vendas", "demo",
            )


def _set_dataset(data, source, label, source_type):
    if data.empty:
        raise ValueError("A base não contém registros de vendas. Envie um arquivo com dados.")
    if data["data"].isna().any():
        raise ValueError("O CSV contém datas ausentes ou inválidas. Revise a coluna data.")
    numeric = data[["quantidade", "preco_unitario", "desconto", "faturamento"]]
    if not numeric.apply(lambda column: column.map(_is_number)).all().all():
        raise ValueError("O CSV contém valores numéricos ausentes ou inválidos. Revise as colunas de vendas.")
    entity_index = build_entity_index(data)
    periods = available_periods(data)
    context = dataset_context(data, source_type)
    if source_type == "olist":
        data.attrs["dataset_context"] = {
            "source_type": "olist", "source_label": context["source_label"],
            "region_label": context["region_label"], "product_label": context["product_label"],
            "seller_label": context["seller_label"], "discount_available": False,
            "revenue_definition": context["revenue_definition"],
            "quantity_definition": context["quantity_definition"], "row_definition": context["row_definition"],
            "limitations": context["ai_context"]["restricoes"],
            **data.attrs.get("dataset_context", {}),
            "coverage": context["monthly_coverage"],
            "missing_category": context["missing_category"],
        }
    st.session_state.data = data
    st.session_state.data_source = source
    st.session_state.data_label = label
    st.session_state.data_source_type = source_type
    st.session_state.entity_index = entity_index
    st.session_state.available_periods = periods
    st.session_state.data_context = context
    st.session_state.messages = []
    st.session_state.last_investigation = None
    st.session_state.last_investigation_feedback = None
    st.session_state.detection_results = None
    st.session_state.detection_feedback = None
    st.session_state.investigation_time = None
    st.session_state.load_feedback = None
    for key in ("historical_result", "historical_alert_date", "segment_result", "historical_segment",
                "review_panel", "review_case_id", "benchmark_result", "benchmark_suite_result", "explanation_comparison"):
        st.session_state.pop(key, None)
    for key in ("compare_previous", "compare_current", "dashboard_dimension", "report_dimension"):
        st.session_state.pop(key, None)


def _load_dataset(loader, source, label, source_type):
    try:
        _set_dataset(loader(), source, label, source_type)
    except Exception as error:
        st.session_state.load_feedback = classify_error(error, context="data")


def _load_olist_data(folder, first_month=None, last_month=None):
    data = _read_local_dataset("olist", input_dir=folder)
    if first_month is None and last_month is None:
        return data
    try:
        first = pd.Period(first_month, freq="M")
        last = pd.Period(last_month, freq="M")
        if str(first) != first_month or str(last) != last_month or first > last:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Informe um intervalo de meses válido no formato AAAA-MM, com início anterior ou igual ao fim.") from None
    months = data["data"].dt.to_period("M")
    selected = data.loc[months.between(first, last)].copy()
    selected.attrs["dataset_context"] = {
        **data.attrs.get("dataset_context", {}),
        "selected_start": str(first), "selected_end": str(last),
        "selection_reason": "Recorte definido explicitamente pelo usuário; o arquivo bruto foi preservado.",
    }
    return selected


def _render_sidebar():
    with st.sidebar:
        st.subheader("Observatório", icon=":material/query_stats:")
        st.caption("Anomalias em dados de vendas")
        with st.container(border=True):
            st.markdown("**Sua base de análise**")
            source = st.selectbox("Origem dos dados", ["Demonstração", "Enviar CSV", "Olist local"], key="data_source_selector")
            if source == "Demonstração":
                scenario = st.selectbox("Cenário", list(DEMO_SCENARIOS), key="demo_scenario")
                path, description = DEMO_SCENARIOS[scenario]
                st.caption(description)
                if st.button("Explorar cenário", type="primary", icon=":material/play_arrow:", width="stretch", key="load_demo"):
                    _load_dataset(lambda: _read_local_dataset("csv", csv_path=str(PROJECT_ROOT / path)), path, scenario, "demo")
            elif source == "Enviar CSV":
                uploaded = st.file_uploader("Arquivo de vendas", type=["csv"], key="uploaded_csv")
                if st.button("Carregar arquivo", icon=":material/upload_file:", type="primary", disabled=uploaded is None, width="stretch", key="load_upload"):
                    _load_dataset(lambda: prepare_sales_data(pd.read_csv(uploaded)), uploaded.name, uploaded.name, "csv")
                st.caption("O formato das colunas está em Dados e relatório.")
            else:
                folder = st.text_input("Pasta da Olist", value=str(PROJECT_ROOT / "data/olist_raw"))
                with st.expander("Recortar por mês (opcional)"):
                    use_range = st.checkbox("Aplicar um intervalo à análise", key="olist_use_range")
                    first_month = st.text_input("Primeiro mês (AAAA-MM)", value="2017-01", disabled=not use_range, key="olist_first_month")
                    last_month = st.text_input("Último mês (AAAA-MM)", value="2018-07", disabled=not use_range, key="olist_last_month")
                    st.caption("O recorte fica identificado no painel. Sem esta opção, todo o histórico é carregado.")
                if st.button("Carregar Olist", icon=":material/folder_open:", type="primary", width="stretch", key="load_olist"):
                    with st.spinner("Preparando os dados..."):
                        label = f"Olist · {first_month} a {last_month}" if use_range else "Olist · histórico completo"
                        _load_dataset(lambda: _load_olist_data(folder, first_month if use_range else None, last_month if use_range else None), folder, label, "olist")
        if st.session_state.load_feedback:
            _render_feedback(st.session_state.load_feedback)
        if st.session_state.data is not None:
            st.badge("Base carregada", color="green", icon=":material/check_circle:")
            st.caption(f"{st.session_state.data_label or Path(st.session_state.data_source).name} · {_format_integer(len(st.session_state.data))} registros")

        st.subheader("Assistência de IA", icon=":material/auto_awesome:")
        try:
            configured = not isinstance(create_llm_provider_from_env(), MockLLMProvider)
        except Exception:
            configured = False
        if "use_ai" not in st.session_state:
            st.session_state.use_ai = configured
        if not configured:
            st.session_state.use_ai = False
        st.toggle("Usar IA na explicação", key="use_ai", disabled=not configured)
        if not st.session_state.use_ai:
            st.badge("Análise local", icon=":material/computer:", color="blue")
        elif st.session_state.ai_status == "unavailable":
            st.badge("IA indisponível · alternativa local", icon=":material/cloud_off:", color="orange")
        elif st.session_state.ai_status == "connected":
            st.badge("IA respondeu à última consulta", icon=":material/cloud_done:", color="green")
        else:
            st.badge("IA configurada", icon=":material/cloud_queue:", color="violet")
        st.caption("Os indicadores e gráficos são calculados na base. A IA ajuda a interpretar perguntas e explicar as evidências.")
        if not configured:
            with st.expander("Ativar a IA"):
                st.markdown("Configure sua chave no arquivo .env do projeto e reinicie o aplicativo. A análise local já está disponível.")
        st.caption(":material/contrast: Tema claro ou escuro no menu do app → Settings → Theme.")


def _render_header():
    title, status = st.columns([3, 1], vertical_alignment="center")
    with title:
        st.caption("VENDAS · INVESTIGAÇÃO DE DADOS")
        st.title("Observatório de anomalias")
        st.markdown("Entenda **o que mudou** nas vendas e quais evidências merecem sua atenção.")
    with status:
        if st.session_state.data is not None:
            is_demo = st.session_state.data_source_type == "demo"
            label = "Dados fictícios" if is_demo else "Base pública · Olist" if st.session_state.data_source_type == "olist" else "Base própria"
            st.badge(label, color="violet" if is_demo else "blue", icon=":material/science:" if is_demo else ":material/database:")
            periods = st.session_state.available_periods or available_periods(st.session_state.data)
            if periods:
                st.caption(f"{period_label(periods[0])} → {period_label(periods[-1])}")


def _render_empty_state():
    with st.container(border=True):
        st.subheader("Sua investigação começa com os dados", icon=":material/database:")
        st.markdown("Escolha um cenário de demonstração ou carregue um CSV na barra lateral. O painel mostra a evolução das vendas e destaca mudanças entre meses.")
        st.code("data,produto,categoria,regiao,vendedor,quantidade,preco_unitario,desconto,faturamento", language="text")


def _render_dataset_context(data):
    if st.session_state.data_source_type != "olist":
        return
    context = st.session_state.data_context or dataset_context(data, "olist")
    st.caption(":material/public: Base real Olist · pedidos entregues · valor dos produtos sem frete · datas de compra. Região corresponde à UF do cliente.")
    with st.expander("Cobertura e limites desta base real", icon=":material/fact_check:"):
        for note in context["notes"]:
            st.markdown(f"- {note}")
        st.caption("As sinalizações desta base não têm gabarito de anomalia ou causa. Elas precisam de validação; contá-las não mede a acurácia do detector.")
        st.markdown("[Fonte pública da Olist no Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)")


def _dimension_options():
    if st.session_state.data_source_type == "olist":
        return {"Produto (ID)": "produto", "Categoria": "categoria", "UF do cliente": "regiao", "Vendedor (ID)": "vendedor"}
    return DIMENSIONS


def _dimension_label(dimension, report):
    context = report.get("contexto_da_base", {})
    keys = {"produto": "product_label", "regiao": "region_label", "vendedor": "seller_label"}
    return context.get(keys.get(dimension), COLUMN_LABELS.get(dimension, dimension))


def _render_data_overview(data, monthly):
    latest = monthly.iloc[-1]
    quantities = sales_by_period(data)["quantidade"].tolist()
    cards = st.columns(4)
    cards[0].metric("Faturamento da base", _format_currency(total_revenue(data)), border=True, height=170, chart_data=monthly["valor"].tolist(), chart_type="area")
    cards[1].metric(
        f"Faturamento · {latest['periodo_label']}", _format_currency(latest["valor"]),
        delta=_format_pct(latest["variacao_percentual"]) if _is_number(latest["variacao_percentual"]) else None,
        delta_description="em relação ao mês anterior", border=True, height=170,
    )
    cards[2].metric("Unidades vendidas", _format_integer(total_quantity(data)), border=True, height=170, chart_data=quantities, chart_type="bar")
    cards[3].metric("Variações sinalizadas", str(int(monthly["anomalia"].sum())), border=True, height=170, help=f"Mudanças mensais de faturamento a partir de {st.session_state.threshold_pct:g}%. Um sinal pede investigação; não prova uma causa.")
    st.caption(f"{st.session_state.data_label or 'Base carregada'} · {_format_integer(len(data))} registros · {len(monthly)} meses observados · sinalização por variação a partir de {st.session_state.threshold_pct:g}%")


def _focus_comparison(data, monthly):
    flagged = monthly.loc[monthly["anomalia"]]
    if not flagged.empty:
        row = flagged.loc[flagged["variacao_percentual"].abs().idxmax()]
        current = row["periodo"]
        return period_comparison(data, str(pd.Period(current, freq="M") - 1), current), True
    return period_comparison(data), False


def _open_investigation(previous, current):
    st.session_state.workspace_view = "Investigação"
    st.session_state.investigation_kind = "Comparar períodos"
    st.session_state.compare_previous = previous
    st.session_state.compare_current = current


def _render_dashboard(data, monthly):
    comparison, flagged = _focus_comparison(data, monthly)
    plot, signal = st.columns([2, 1], gap="medium")
    with plot, st.container(border=True):
        st.subheader("Evolução do faturamento", icon=":material/monitoring:")
        st.caption("Passe o cursor para ver os valores. Pontos em âmbar destacam variações que atingem o limiar.")
        st.altair_chart(trend_chart(monthly), width="stretch")
    with signal, st.container(border=True):
        st.markdown("**No radar**")
        if comparison["available"]:
            st.badge("Mudança para investigar" if flagged else "Última comparação", color="orange" if flagged else "blue", icon=":material/travel_explore:")
            st.subheader(_format_pct(comparison["variacao_percentual_faturamento"]))
            st.markdown(f"**{period_label(comparison['periodo_anterior'])} → {period_label(comparison['periodo_atual'])}**")
            st.caption(f"De {_format_currency(comparison['faturamento_anterior'])} para {_format_currency(comparison['faturamento_atual'])}.")
            drivers = dimension_contributions(data, comparison["periodo_anterior"], comparison["periodo_atual"], limit=1)
            if not drivers.empty and drivers.iloc[0]["variacao_faturamento"] != 0:
                driver = drivers.iloc[0]
                st.markdown(f"Maior mudança por produto: **{driver['segmento']}**")
                st.caption(f"Diferença de {_format_currency(driver['variacao_faturamento'])}. Abra a investigação para comparar as evidências.")
            st.button("Investigar esta mudança", type="primary", icon=":material/arrow_forward:", width="stretch", key="investigate_focus", on_click=_open_investigation, args=(comparison["periodo_anterior"], comparison["periodo_atual"]))
        else:
            st.badge("Sem comparação mensal", color="gray", icon=":material/calendar_month:")
            st.markdown("Precisamos de dois meses consecutivos para destacar uma mudança neste painel.")
            st.caption("Você pode escolher dois meses disponíveis na área de investigação.")
    _render_history_note(monthly)

    impacts, changes = st.columns([1.35, 1], gap="medium")
    with impacts, st.container(border=True):
        st.subheader("Onde a mudança se concentra", icon=":material/account_tree:")
        dimensions = _dimension_options()
        dimension_label = st.segmented_control("Comparar por", list(dimensions), default=next(iter(dimensions)) if "dashboard_dimension" not in st.session_state else None, required=True, key="dashboard_dimension", width="stretch")
        if comparison["available"]:
            frame = dimension_contributions(data, comparison["periodo_anterior"], comparison["periodo_atual"], dimensions[dimension_label])
            st.altair_chart(contribution_chart(frame, period_label(comparison["periodo_anterior"]), period_label(comparison["periodo_atual"])), width="stretch")
            st.caption("Vermelho: redução de faturamento. Verde: aumento. Cada barra mostra a diferença de um segmento; até 8 segmentos com maior mudança absoluta.")
        else:
            st.info("Carregue pelo menos dois meses ou abra a investigação para escolher uma comparação.")
    with changes, st.container(border=True):
        st.subheader("Ritmo das vendas", icon=":material/swap_vert:")
        st.caption("Variação percentual do faturamento em relação ao mês anterior.")
        if monthly["percentual_disponivel"].any():
            st.altair_chart(change_chart(monthly), width="stretch")
        else:
            st.info("Ainda não há uma comparação mensal com percentual calculável.")
    with st.expander("Como interpretar este painel", icon=":material/help:"):
        st.markdown(
            "**Sinal de anomalia:** a mudança do faturamento entre dois meses consecutivos ultrapassa o limiar escolhido. "
            "Esse sinal ajuda a priorizar a investigação.\n\n"
            "**Evidências:** compare produtos, regiões, categorias e vendedores separadamente, além de quantidade, preço e desconto. "
            "Os mesmos registros aparecem nessas dimensões; não some suas contribuições.\n\n"
            "**Limites:** associações nos dados não demonstram causalidade. Meses sem registros não são tratados como faturamento zero."
        )


def _render_history_note(monthly):
    if len(monthly) < 3:
        st.caption(":material/info: Histórico curto: a sinalização usa variação percentual. Com estes dados não é possível avaliar sazonalidade ou um padrão de longo prazo.")
    if len(monthly) > 1 and not monthly["comparavel"].iloc[1:].all():
        st.caption(":material/event_busy: Há meses sem registros no intervalo. O painel não calcula variação mensal através dessas lacunas.")



def _render_investigation(data):
    st.subheader("Da mudança às evidências", icon=":material/troubleshoot:")
    st.caption("Detecte uma variação ou escolha dois períodos. A investigação compara produtos, categorias, regiões e vendedores.")
    periods = available_periods(data)
    if len(periods) < 2:
        _render_feedback(classify_error(ValueError("Precisamos de dois periodos para comparar")))
        return
    kind = st.segmented_control("Como investigar", ["Detectar automaticamente", "Comparar períodos"], default="Detectar automaticamente" if "investigation_kind" not in st.session_state else None, required=True, key="investigation_kind", width="stretch")
    if kind == "Comparar períodos":
        with st.form("manual_investigation"):
            before, after = st.columns(2)
            previous = before.selectbox("Período de referência", periods, index=None if "compare_previous" in st.session_state else max(len(periods) - 2, 0), format_func=period_label, key="compare_previous")
            current = after.selectbox("Período em análise", periods, index=None if "compare_current" in st.session_state else len(periods) - 1, format_func=period_label, key="compare_current")
            st.caption("A comparação detalha faturamento, quantidade, preço unitário médio e desconto médio.")
            submitted = st.form_submit_button("Investigar comparação", type="primary", icon=":material/travel_explore:", width="stretch")
        if submitted:
            comparison = period_comparison(data, previous, current)
            if not comparison["available"]:
                st.warning("Escolha como referência um mês anterior ao período em análise.", icon=":material/calendar_month:")
            else:
                _execute_investigation(data, previous, current)
    else:
        with st.form("automatic_investigation"):
            col1, col2 = st.columns(2)
            metric_label = col1.selectbox("Indicador", ["Faturamento", "Quantidade vendida"], key="detection_metric")
            threshold = col2.number_input("Sinalizar variações a partir de (%)", min_value=0.0, max_value=10000.0, value=st.session_state.threshold_pct, step=5.0, key="threshold_input")
            with st.expander("Critérios avançados", icon=":material/tune:"):
                method_labels = {"Variação percentual": "percentual", "Combinado": "combinado", "Z-score": "zscore", "Intervalo interquartil (IQR)": "iqr", "Média móvel": "media_movel"}
                method_label = st.selectbox("Método de detecção", list(method_labels), key="detection_method")
                min_value = st.number_input("Valor mínimo nos dois períodos", value=0.0, min_value=0.0, help="Ignora comparações em que um dos períodos tenha menos que este valor.", key="detection_minimum")
                st.caption("Z-score requer ao menos 3 meses; IQR, 5 meses; média móvel, 4 meses. O método combinado utiliza os critérios com histórico disponível.")
            if metric_label == "Quantidade vendida":
                st.caption("O sinal será buscado em quantidade; a investigação detalha também as mudanças de faturamento.")
            submitted = st.form_submit_button("Detectar e investigar", type="primary", icon=":material/search_check:", width="stretch")
        if submitted:
            method = method_labels[method_label]
            needed = {"zscore": 3, "iqr": 5, "media_movel": 4}.get(method, 2)
            if len(periods) < needed:
                st.info(f"Este método precisa de pelo menos {needed} meses; a base tem {len(periods)}. Escolha Variação percentual para investigar este histórico.", icon=":material/info:")
            else:
                _execute_investigation(data, metric="faturamento" if metric_label == "Faturamento" else "quantidade", method=method, threshold_pct=threshold, min_period_value=min_value)
    if st.session_state.detection_feedback:
        _render_feedback(st.session_state.detection_feedback)
    if st.session_state.detection_results is not None:
        with st.expander("Ver períodos avaliados", icon=":material/checklist:"):
            rows = [{"Referência": period_label(item["periodo_anterior"]), "Em análise": period_label(item["periodo_atual"]), "Variação": _format_pct(item["variacao_percentual"]), "Resultado": "Sinalizado" if item["anomalia"] else "Dentro dos critérios"} for item in st.session_state.detection_results]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if st.session_state.last_investigation is not None:
        _render_investigation_result(data)


def _execute_investigation(data, previous=None, current=None, **settings):
    st.session_state.detection_feedback = None
    st.session_state.detection_results = None
    st.session_state.last_investigation = None
    with st.status("Comparando períodos e reunindo evidências...", expanded=False) as status:
        try:
            if previous is not None:
                description = f"Investigue a variação de faturamento de {previous} para {current}."
                _perform_investigation(description, data)
            else:
                _run_auto_investigation(data, **settings)
            status.update(label="Evidências prontas para análise", state="complete")
        except Exception as error:
            feedback = classify_error(error)
            st.session_state.detection_feedback = feedback
            status.update(label=feedback["title"], state="complete" if feedback["kind"] == "info" else "error")
    st.rerun()


class _ResilientProvider:
    """Preserve deterministic evidence when an optional model is unavailable."""

    def __init__(self, provider, remote=False, feedback=None):
        self.provider = provider
        self.remote = remote
        self.feedback = feedback
        self.used_ai = False

    def generate(self, system_prompt, user_prompt, context):
        try:
            result = self.provider.generate(system_prompt, user_prompt, context)
            self.used_ai = self.remote
            return result
        except Exception as error:
            self.feedback = classify_error(error, context="api")
            return MockLLMProvider().generate(system_prompt, user_prompt, context)

    def interpret_query(self, question, context):
        try:
            result = self.provider.interpret_query(question, context)
            self.used_ai = self.remote and result is not None
            return result
        except Exception as error:
            self.feedback = classify_error(error, context="api")
            return None


def _provider_for_request():
    if not st.session_state.get("use_ai", False):
        return _ResilientProvider(MockLLMProvider())
    try:
        provider = create_llm_provider_from_env()
        return _ResilientProvider(provider, remote=not isinstance(provider, MockLLMProvider))
    except Exception as error:
        return _ResilientProvider(MockLLMProvider(), feedback=classify_error(error, context="api"))


def _record_provider_status(provider):
    if provider.feedback:
        st.session_state.ai_status = "unavailable"
        st.session_state.last_api_feedback = provider.feedback
    elif provider.used_ai:
        st.session_state.ai_status = "connected"
        st.session_state.last_api_feedback = None


def _perform_investigation(description, data, extra_evidence=None):
    provider = _provider_for_request()
    investigation = InvestigationAgent(llm_provider=provider).investigate(description, data, extra_evidence=extra_evidence)
    _record_provider_status(provider)
    st.session_state.last_investigation = investigation
    st.session_state.last_investigation_feedback = provider.feedback
    st.session_state.last_investigation_mode = "ai" if provider.used_ai else "local"
    st.session_state.investigation_time = datetime.now().strftime("%d/%m/%Y às %H:%M")
    return investigation


def _run_auto_investigation(data, metric="faturamento", method="percentual", threshold_pct=None, min_period_value=0.0):
    anomalies = detect_period_anomalies(data, metric=metric, method=method, threshold_pct=st.session_state.threshold_pct if threshold_pct is None else threshold_pct, min_period_value=min_period_value)
    anomalies = [item for item in anomalies if pd.Period(item["periodo_atual"], freq="M") == pd.Period(item["periodo_anterior"], freq="M") + 1]
    if not anomalies:
        raise ValueError("Precisamos de dois periodos consecutivos para detectar uma variação mensal.")
    st.session_state.detection_results = anomalies
    anomaly = get_most_relevant_anomaly(anomalies)
    if anomaly is None:
        raise ValueError("Nenhuma anomalia foi detectada com os parâmetros atuais.")
    return _perform_investigation(build_anomaly_description(anomaly), data, extra_evidence=anomaly)


def _render_feedback(feedback, local_results=False):
    tone = "warning" if local_results else feedback.get("kind", "warning")
    renderer = {"error": st.error, "warning": st.warning, "info": st.info}.get(tone, st.info)
    renderer(f"**{feedback['title']}**\n\n{feedback['message']}\n\n**O que fazer:** {feedback['action']}", icon=":material/info:" if tone == "info" else ":material/warning:")
    if local_results:
        st.caption(":material/check_circle: A comparação e as evidências abaixo foram concluídas com a análise local.")
    if feedback.get("technical_details"):
        with st.expander("Detalhes técnicos", icon=":material/code:"):
            st.code(sanitize_details(feedback["technical_details"]), language="text")


def _local_conclusion(report):
    summary = report.get("resumo_comparativo", {})
    if not summary:
        return "A investigação foi concluída com os dados disponíveis. Escolha dois períodos para obter uma comparação detalhada."
    periods = report.get("periodos", {})
    delta = summary.get("variacao_faturamento", 0)
    direction = "caiu" if delta < 0 else "aumentou" if delta > 0 else "não variou"
    text = (
        f"Entre **{period_label(periods.get('periodo_anterior'))}** e **{period_label(periods.get('periodo_atual'))}**, "
        f"o faturamento **{direction}**, de **{_format_currency(summary.get('faturamento_anterior'))}** "
        f"para **{_format_currency(summary.get('faturamento_atual'))}** "
        f"(variação de **{_format_pct(summary.get('variacao_percentual_faturamento'))}**)."
    )
    causes = report.get("causas_provaveis", [])
    if causes:
        top = causes[0]
        metrics = top.get("metricas", {})
        text += (
            f"\n\nA evidência priorizada está associada a **{top['valor']}** "
            f"na dimensão **{_dimension_label(top['dimensao'], report).lower()}**. "
            f"Esse segmento apresentou diferença de **{_format_currency(metrics.get('variacao_faturamento'))}** "
            f"e de **{_format_integer(metrics.get('variacao_quantidade', 0))} unidades**."
        )
    return text


def _render_investigation_result(data):
    investigation = st.session_state.last_investigation
    report = investigation.structured_report
    mode = st.session_state.last_investigation_mode
    st.subheader("Resultado da investigação", icon=":material/fact_check:")
    if st.session_state.last_investigation_feedback:
        _render_feedback(st.session_state.last_investigation_feedback, local_results=True)
    st.badge("Evidências locais · explicação da IA" if mode == "ai" else "Investigação com análise local", color="violet" if mode == "ai" else "blue", icon=":material/auto_awesome:" if mode == "ai" else ":material/computer:")
    st.markdown(investigation.final_conclusion if mode == "ai" else _local_conclusion(report))
    _render_comparison_metrics(report)
    render_revenue_explanation(report)
    plot, evidence = st.columns([1, 1.15], gap="medium")
    with plot, st.container(border=True):
        st.markdown("**Antes e depois**")
        st.altair_chart(comparison_chart(report), width="stretch")
    with evidence, st.container(border=True):
        st.markdown("**Evidências prioritárias**")
        _render_cause_cards(report)
    periods = report.get("periodos", {})
    if periods.get("periodo_anterior") and periods.get("periodo_atual"):
        with st.container(border=True):
            st.markdown("**Explore a contribuição de cada segmento**")
            dimensions = _dimension_options()
            label = st.segmented_control("Dimensão", list(dimensions), default=next(iter(dimensions)) if "report_dimension" not in st.session_state else None, required=True, key="report_dimension", width="stretch")
            frame = dimension_contributions(data, periods["periodo_anterior"], periods["periodo_atual"], dimensions[label])
            st.altair_chart(contribution_chart(frame, period_label(periods["periodo_anterior"]), period_label(periods["periodo_atual"])), width="stretch")
            st.caption("As dimensões compartilham registros. As contribuições de produtos, regiões, categorias e vendedores não devem ser somadas entre si.")
    st.caption("As evidências indicam associações nos dados. Estoque, campanhas e sazonalidade precisam de informações adicionais para confirmar uma causa.")


def _render_comparison_metrics(report):
    summary = report.get("resumo_comparativo", {})
    if not summary:
        return
    cards = st.columns(3)
    cards[0].metric("Diferença de faturamento", _format_currency(summary.get("variacao_faturamento")), border=True)
    cards[1].metric("Variação de unidades", _format_integer(summary.get("variacao_quantidade", 0)), delta=_format_pct(summary.get("variacao_percentual_quantidade")) if _is_number(summary.get("variacao_percentual_quantidade")) else None, border=True)
    discount = summary.get("variacao_desconto_medio")
    discount_available = report.get("contexto_da_base", {}).get("discount_available", True)
    value = "Não disponível" if not discount_available else f"{_format_decimal(discount * 100, 1)} p.p." if _is_number(discount) else "—"
    cards[2].metric("Mudança no desconto médio", value, border=True, help="O desconto não é fornecido pela fonte Olist." if not discount_available else "Diferença em pontos percentuais. O desconto médio é calculado por registro de venda.")


def _render_cause_cards(report):
    causes = report.get("causas_provaveis", [])
    if not causes:
        st.caption("Não houve evidência suficiente para priorizar um segmento.")
        return
    for index, cause in enumerate(causes[:3], start=1):
        metrics = cause.get("metricas", {})
        st.markdown(f"**{index:02d} · {cause['valor']}** · {_dimension_label(cause['dimensao'], report)}")
        st.caption(f"Faturamento: {_format_currency(metrics.get('variacao_faturamento'))} · Quantidade: {_format_integer(metrics.get('variacao_quantidade', 0))} un.")
        price, discount = metrics.get("variacao_preco_unitario_medio"), metrics.get("variacao_desconto_medio")
        changes = []
        if _is_number(price) and abs(price) > 0.001:
            changes.append(f"Preço médio: {_format_currency(price)}")
        if report.get("contexto_da_base", {}).get("discount_available", True) and _is_number(discount) and abs(discount) > 0.00001:
            changes.append(f"Desconto médio: {_format_decimal(discount * 100, 1)} p.p.")
        if changes:
            st.caption(" · ".join(changes))
    st.caption("Ordem definida por um ranking de impacto. Ela não representa uma probabilidade causal.")



def _render_chat(data):
    header, action = st.columns([3, 1], vertical_alignment="center")
    with header:
        st.subheader("Converse sobre a anomalia", icon=":material/forum:")
        st.caption("Peça uma investigação, compare períodos ou confira as evidências de uma mudança.")
    with action:
        if st.button("Nova conversa", icon=":material/add_comment:", key="clear_chat", disabled=not st.session_state.messages):
            st.session_state.messages = []
    periods = available_periods(data)
    suggested_prompt = None
    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("**Comece pela mudança que você quer entender**")
            if len(periods) >= 2:
                before, after = periods[-2:]
                suggestions = [
                    ("Detectar uma anomalia", "Encontre uma anomalia no faturamento das vendas."),
                    ("Investigar a variação", f"Investigue a variação do faturamento entre {before} e {after}."),
                    ("Ver a evolução mensal", "Como o faturamento evoluiu por mês?"),
                ]
            else:
                suggestions = [("Ver produtos", "Quais produtos tiveram maior faturamento?"), ("Ver regiões", "Qual foi o faturamento por região?")]
            with st.container(horizontal=True):
                for index, (label, question) in enumerate(suggestions):
                    if st.button(label, key=f"chat_suggestion_{index}", icon=":material/north_east:"):
                        suggested_prompt = question
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            _render_chat_message(message)
    prompt = st.chat_input("Ex.: Investigue a queda de faturamento entre dois meses…", key="chat_prompt", submit_mode="disable") or suggested_prompt
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Conferindo os dados e as evidências..."):
            response = _handle_user_prompt(prompt, data)
        _render_chat_message(response)
    st.session_state.messages.append({"role": "assistant", **response})
    st.rerun()


def _handle_user_prompt(prompt, data):
    normalized = normalize_text(prompt)
    auto_request = "anomalia" in normalized and any(term in normalized for term in ("procure", "detectar", "detecte", "encontre", "identifique", "busque"))
    if auto_request:
        try:
            return _investigation_message(_run_auto_investigation(data))
        except Exception as error:
            return {"content": "Confira o resultado da detecção abaixo.", "feedback": classify_error(error)}
    periods = extract_periods_from_text(prompt, data)
    investigation_request = any(term in normalized for term in ("por que", "porque", "causa", "motivo", "explique", "investigue", "responsavel", "anomalia", "queda", "caiu", "aumentou"))
    if investigation_request:
        if len(periods) < 2:
            available = available_periods(data)
            example = f"Investigue a variação entre {available[-2]} e {available[-1]}." if len(available) >= 2 else "Carregue uma base com pelo menos dois meses."
            return {"content": f"Para investigar essa mudança, preciso de dois períodos de comparação.\n\n**Exemplo com sua base:** {example}", "mode": "local"}
        try:
            return _investigation_message(_perform_investigation(prompt, data))
        except Exception as error:
            return {"content": "Não foi possível concluir esta investigação.", "feedback": classify_error(error)}
    provider = _provider_for_request()
    source_context = data.attrs.get("dataset_context", {})
    if source_context.get("discount_available") is False and "desconto" in normalized:
        return {"content": "A base Olist não fornece o desconto aplicado às vendas. O campo zero é preenchido pelo adaptador e não permite concluir que as vendas ocorreram sem desconto. Posso investigar faturamento, quantidade e preço dos itens.", "mode": "local"}
    context = {
        "colunas": list(data.columns), "periodos_disponiveis": available_periods(data),
        "metricas_suportadas": ["faturamento", "quantidade", "preco_unitario"] + (["desconto"] if source_context.get("discount_available", True) else []),
        "contexto_da_fonte": source_context,
        "limites_da_base": (st.session_state.data_context or {}).get("ai_context", {}),
        "dimensoes_suportadas": ["produto", "categoria", "regiao", "vendedor", "periodo"],
        "conversa_recente": [{"role": item["role"], "content": item["content"][:1000]} for item in st.session_state.messages[:-1][-4:]],
    }
    plan = provider.interpret_query(prompt, context)
    _record_provider_status(provider)
    answer = answer_sales_question(prompt, data, entity_index=st.session_state.entity_index, query_plan=plan)
    if answer is None:
        answer = {"content": "Posso investigar variações de faturamento, comparar períodos e detalhar quantidade, preço e desconto por produto, categoria, região ou vendedor. Inclua dois meses para começar uma investigação."}
    return {**answer, "feedback": provider.feedback, "mode": "ai" if provider.used_ai else "local"}


def _investigation_message(investigation):
    mode = st.session_state.last_investigation_mode
    return {
        "content": investigation.final_conclusion if mode == "ai" else _local_conclusion(investigation.structured_report),
        "report": investigation.structured_report, "feedback": st.session_state.last_investigation_feedback, "mode": mode,
    }


def _render_chat_message(message):
    if message.get("feedback"):
        _render_feedback(message["feedback"], local_results=message.get("mode") == "local" and bool(message.get("report") or message.get("tables")))
    if message.get("mode"):
        st.caption(":material/auto_awesome: IA na interpretação · cálculos locais" if message["mode"] == "ai" else ":material/computer: Resposta pela análise local")
    st.markdown(message["content"])
    if message.get("report"):
        report = message["report"]
        _render_comparison_metrics(report)
        st.altair_chart(comparison_chart(report), width="stretch")
        with st.expander("Ver evidências prioritárias", icon=":material/fact_check:"):
            _render_cause_cards(report)
    for table in message.get("tables", []):
        frame = pd.DataFrame(table.get("data", []))
        if frame.empty:
            continue
        with st.container(border=True):
            st.markdown(f"**{table['title']}**")
            chart = table.get("chart")
            friendly = frame.rename(columns=COLUMN_LABELS)
            if chart and chart["x"] in frame and chart["y"] in frame:
                x, y = COLUMN_LABELS.get(chart["x"], chart["x"]), COLUMN_LABELS.get(chart["y"], chart["y"])
                if chart["type"] == "line":
                    st.line_chart(friendly, x=x, y=y, height=240)
                else:
                    st.bar_chart(friendly, x=x, y=y, horizontal=True, height=240)
            with st.expander("Ver valores da resposta"):
                st.dataframe(friendly, width="stretch", hide_index=True)


def _render_data_and_report(data):
    st.subheader("Dados que sustentam a investigação", icon=":material/database:")
    st.caption(f"Fonte: {st.session_state.data_label or st.session_state.data_source} · exibindo até 100 registros")
    frame = data.head(100).rename(columns=COLUMN_LABELS)
    if st.session_state.data_source_type == "olist":
        frame = frame.rename(columns={"Produto": "ID do produto", "Região": "UF do cliente", "Vendedor": "ID do vendedor"})
    discount_available = data.attrs.get("dataset_context", {}).get("discount_available", True)
    formatted = frame.style.format({"Faturamento": _format_currency, "Preço unitário": _format_currency, "Desconto": lambda value: f"{value * 100:.1f}%" if discount_available else "Não disponível", "Data": lambda value: value.strftime("%d/%m/%Y")})
    st.dataframe(formatted, width="stretch", hide_index=True, height=340)
    st.download_button("Baixar base padronizada", data=data.to_csv(index=False).encode("utf-8-sig"), file_name="vendas_padronizadas.csv", mime="text/csv", icon=":material/download:", key="download_data")
    with st.expander("Formato do CSV e critérios da análise", icon=":material/schema:"):
        st.code("data,produto,categoria,regiao,vendedor,quantidade,preco_unitario,desconto,faturamento", language="text")
        st.markdown("Datas no formato AAAA-MM-DD; quantidade, preço, desconto e faturamento devem ser numéricos. O desconto é uma fração: 0,10 representa 10%. Preço e desconto médios são médias por registro.")
        st.caption("O faturamento é lido da coluna da base. A detecção compara períodos, e as hipóteses são priorizadas pelas evidências calculadas.")
    investigation = st.session_state.last_investigation
    st.subheader("Relatório da investigação", icon=":material/description:")
    if investigation is None:
        st.info("O relatório aparecerá depois de investigar uma mudança. Abra Investigação ou faça uma pergunta ao assistente.", icon=":material/info:")
        return
    st.caption(f"Última investigação · {st.session_state.investigation_time or 'sessão atual'}")
    report = investigation.structured_report
    mode = st.session_state.last_investigation_mode
    conclusion = investigation.final_conclusion if mode == "ai" else _local_conclusion(report)
    with st.container(border=True):
        st.markdown(conclusion)
    filename = f"investigacao_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with st.container(horizontal=True):
        st.download_button("Baixar relatório", data=_report_markdown(investigation, mode), file_name=filename + ".md", mime="text/markdown", icon=":material/download:", key="download_report")
        st.download_button("Baixar registro técnico", data=json.dumps(investigation.to_dict(), ensure_ascii=False, indent=2), file_name=filename + ".json", mime="application/json", icon=":material/data_object:", key="download_log")
    with st.expander("Registro técnico completo", icon=":material/code:"):
        st.json(investigation.to_dict())


def _report_markdown(investigation, mode):
    report = investigation.structured_report
    conclusion = investigation.final_conclusion if mode == "ai" else _local_conclusion(report)
    parts = ["# Investigação de anomalias em vendas", f"Base: {st.session_state.data_label}", f"Modo: {'explicação da IA e cálculos locais' if mode == 'ai' else 'análise local'}", "## Conclusão", conclusion, "## Evidências"]
    for cause in report.get("causas_provaveis", []):
        metrics = cause.get("metricas", {})
        parts.append(f"- {cause['valor']} ({_dimension_label(cause['dimensao'], report)}): variação de {_format_currency(metrics.get('variacao_faturamento'))} e {_format_integer(metrics.get('variacao_quantidade', 0))} unidades.")
    if report.get("contexto_da_base", {}).get("limitations"):
        parts.extend(["## Definição da base", "\n".join(f"- {item}" for item in report["contexto_da_base"]["limitations"])])
    decomposition = report.get("decomposicao_faturamento", {})
    if decomposition.get("available"):
        parts.extend(["## Decomposição do faturamento", decomposition["convention"]])
        parts.extend(f"- {item['label']}: {_format_currency(item['value'])}" for item in decomposition["components"])
        parts.append(f"Erro de reconciliação: {decomposition['reconciliation_error']:.8f}")
    operations = report.get("indicadores_operacionais", {})
    if operations.get("available"):
        parts.extend(["## Indicadores operacionais", json.dumps(operations["monthly"], ensure_ascii=False, indent=2)])
    parts.extend(["## Limitações da investigação", "\n".join(f"- {item}" for item in report.get("limitacoes", []))])
    parts.extend(["## Limites", "As evidências mostram associações e não confirmam causalidade. O ranking prioriza impacto nos dados; não representa uma probabilidade causal."])
    return "\n\n".join(parts)


def _is_number(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _format_decimal(value, precision=2):
    if not _is_number(value):
        return "—"
    return f"{float(value):,.{precision}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _format_currency(value):
    return f"R$ {_format_decimal(value)}" if _is_number(value) else "—"


def _format_integer(value):
    return f"{int(value):,}".replace(",", ".") if _is_number(value) else "—"


def _format_pct(value):
    return f"{_format_decimal(value, 1)}%" if _is_number(value) else "Sem base percentual"



