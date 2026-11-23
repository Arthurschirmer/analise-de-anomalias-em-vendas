"""Demonstração offline de anomalias condicionais ao calendário conhecido."""
from pathlib import Path
from functools import partial

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluation.contextual_benchmark import run_contextual_benchmark, run_contextual_fixture
from src.evaluation.pdf_report import build_alert_pdf
from src.analysis.alert_explanation import explain_alert
from src.agent.alert_conversation import build_alert_context


ROOT = Path(__file__).resolve().parents[2]
METHODS = {"historical_without_context": "Sem calendário", "contextual": "Com calendário"}
CONTEXT_LABELS = {"normal": "Dia sem promoção", "promotion": "Promoção",
                  "unknown": "Calendário indisponível"}


@st.cache_data(ttl=600, max_entries=12)
def _read_points(path, modified_ns):
    return pd.read_csv(path, parse_dates=["data"])


def evaluate_loaded_fixture(input_dir):
    return run_contextual_fixture(input_dir, ROOT / "outputs/contextual/loaded_demo")


def context_comparison_chart(points):
    """Exibe observação, referência e limites de um único método/execução."""
    frame = points.copy()
    base = alt.Chart(frame).encode(x=alt.X("data:T", title="Dia"))
    band = base.mark_area(color="#6357D9", opacity=.10).encode(
        y=alt.Y("lower:Q", title="Faturamento (R$)"), y2="upper:Q")
    observed = base.mark_line(color="#268D80", strokeWidth=2).encode(
        y="valor:Q", tooltip=[alt.Tooltip("data:T", title="Dia"),
                              alt.Tooltip("valor:Q", title="Vendas (R$)", format=",.2f"),
                              alt.Tooltip("context_label:N", title="Calendário")])
    reference = base.mark_line(color="#6357D9", strokeDash=[5, 4]).encode(y="expected:Q")
    alerts = base.transform_filter(alt.datum.flag == True).mark_point(
        color="#D95C5C", filled=True, size=85).encode(
            y="valor:Q", tooltip=[alt.Tooltip("data:T", title="Dia"),
                                  alt.Tooltip("valor:Q", title="Vendas (R$)", format=",.2f"),
                                  alt.Tooltip("context_label:N", title="Calendário")])
    return (band + observed + reference + alerts).properties(height=320).configure_view(strokeWidth=0)


def _seeds(text):
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError:
        raise ValueError("Informe sementes numéricas separadas por vírgula.") from None
    if not 1 <= len(values) <= 5 or len(set(values)) != len(values):
        raise ValueError("Informe de 1 a 5 sementes distintas.")
    if any(value < 0 or value >= 2**32 for value in values):
        raise ValueError("Cada semente deve estar entre 0 e 4294967295.")
    return values


def _metrics(result):
    table = pd.DataFrame(result["aggregate_metrics"])
    if table.empty:
        st.warning("Não foi possível comparar os métodos neste experimento.")
        return
    table["detector"] = table["detector"].map(METHODS).fillna(table["detector"])
    columns = ["detector", "recall_mean", "precision_mean", "fp_mean", "expected_context_alerts_mean"]
    columns = [column for column in columns if column in table]
    table = table[columns].rename(columns={
        "detector": "Método", "recall_mean": "Anomalias encontradas",
        "precision_mean": "Alertas corretos", "fp_mean": "Alertas indevidos",
        "expected_context_alerts_mean": "Alertas em variações previstas"})
    formats = {name: "{:.1%}" for name in ("Anomalias encontradas", "Alertas corretos") if name in table}
    formats.update({name: "{:.1f}" for name in ("Alertas indevidos", "Alertas em variações previstas") if name in table})
    st.dataframe(table.style.format(formats, na_rep="—"), hide_index=True, width="stretch")
    st.caption("Médias entre as execuções. O alvo são as quedas extras inesperadas; mudanças previstas pelo calendário são consideradas esperadas nesta simulação.")
    st.caption("Anomalias encontradas: parcela das quedas extras identificadas. Alertas corretos: parcela dos alertas que correspondem a essas quedas.")


def _money(value):
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _inspect_day(points, *, show_truth=True, prefer_alert=False):
    options = points["data"].tolist()
    if not options:
        return
    if st.session_state.get("context_selected_day") not in options:
        st.session_state.pop("context_selected_day", None)
    first_index = 0
    if prefer_alert:
        alerted = points.loc[points["eligible"].fillna(False) & points["flag"].fillna(False), "data"]
        if not alerted.empty:
            first_index = options.index(alerted.iloc[0])
    context_by_day = dict(zip(points["data"], points["context_label"]))
    selected = st.selectbox("Examinar um dia do teste" if show_truth else "Entender um dia", options,
                            index=None if "context_selected_day" in st.session_state else first_index,
                            format_func=lambda value: value.strftime("%d/%m/%Y") + " · " + str(context_by_day[value]),
                            key="context_selected_day")
    day = points.loc[points["data"].eq(selected)].iloc[0]
    with st.container(border=True):
        st.markdown("**Calendário informado:** " + str(day["context_label"]))
        if pd.notna(day["valor"]):
            st.markdown(f"Vendas observadas: **{_money(day['valor'])}**.")
        else:
            st.markdown("**Vendas observadas:** informação ausente.")
        if pd.notna(day["expected"]):
            st.markdown(f"Referência para este dia: **{_money(day['expected'])}**.")
        if pd.notna(day["lower"]) and pd.notna(day["upper"]):
            st.caption(f"Faixa de referência usada na análise: {_money(day['lower'])} a {_money(day['upper'])}.")
        if not bool(day["eligible"]):
            st.info("Não há informação suficiente para decidir sobre este dia. Ele fica sem classificação.")
        elif bool(day["flag"]):
            direction = "abaixo" if day["valor"] < day["expected"] else "acima"
            st.warning(f"As vendas ficaram {direction} da faixa de referência. Este dia merece investigação.")
        else:
            st.success("As vendas ficaram dentro da faixa de referência. Este método não emitiu alerta para o dia.")
        st.caption("A promoção é uma informação do calendário. Ela ajuda a interpretar o resultado; confirmar uma causa em dados reais pode exigir outras informações da empresa.")
        if not show_truth and bool(day["eligible"]) and bool(day["flag"]) and st.session_state.get("data") is not None:
            facts = explain_alert(day, {"threshold": day.get("decision_threshold")}, "faturamento")
            facts.update(start_date=selected.date().isoformat(), end_date=selected.date().isoformat(),
                         frequency="D", calibration_method="referência com calendário; limite definido na calibração",
                         calendar_context=str(day["context_label"]))
            context = build_alert_context(st.session_state.data, facts, {})
            context["source"].setdefault("source_label", st.session_state.get("data_label") or "Base fictícia com calendário")
            st.download_button("Baixar alerta em PDF", partial(build_alert_pdf, context),
                               file_name="alerta_" + facts["period"] + ".pdf", mime="application/pdf",
                               key="context_alert_pdf", on_click="ignore", icon=":material/picture_as_pdf:")
            if st.button("Perguntar sobre este alerta", key="context_chat_about_alert", icon=":material/forum:", type="primary"):
                st.session_state.alert_chat_context = context
                st.session_state.monthly_chat_context = None
                st.session_state.messages = []
                st.session_state.workspace_view = "Assistente"
                st.rerun()
        if show_truth:
            with st.expander("Conferir o gabarito separado da simulação"):
                st.markdown("**Queda extra inesperada programada:** " + ("Sim" if bool(day["is_anomaly"]) else "Não"))
                st.caption("O gabarito só é consultado depois da detecção para medir os resultados.")


def render_loaded_context_analysis():
    """Present the loaded calendar-aware analysis without experiment answer keys."""
    st.subheader("O que merece sua atenção", icon=":material/travel_explore:")
    st.markdown("O sistema compara as vendas de cada dia com uma referência que considera o dia da semana e as promoções informadas. Os alertas indicam dias que merecem uma verificação mais detalhada.")
    fixture = st.session_state.get("context_fixture_dir")
    if not fixture:
        st.info("Carregue a base de promoções e quedas na barra lateral para iniciar esta análise.")
        return
    result = st.session_state.get("context_experiment_result")
    if st.button("Analisar novamente" if result else "Analisar esta base", icon=":material/search_check:",
                 key="reanalyze_loaded_context", type="primary"):
        try:
            with st.spinner("Analisando os arquivos da base carregada..."):
                result = evaluate_loaded_fixture(fixture)
            st.session_state.context_experiment_result = result
        except (ValueError, OSError) as error:
            st.warning(f"Não foi possível concluir a análise. Confira os arquivos da base: {error}")
            return
    if result is None:
        st.info("Clique em Analisar esta base para ver os dias sinalizados.")
        return
    try:
        path = Path(result["outputs"]["point_results.csv"])
        all_points = _read_points(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        st.warning("Os resultados não estão disponíveis. Clique em Analisar novamente para recalculá-los.")
        return
    seed = result["seeds"][0]
    selected_run = all_points.loc[all_points["seed"].eq(seed)]
    points = selected_run.loc[selected_run["detector"].eq("contextual")
                              & selected_run["split"].eq("test")].copy().sort_values("data")
    if points.empty:
        st.warning("Esta base ainda não tem informação suficiente para a análise com calendário.")
        return
    # The main analysis deliberately has no access to answer-key classifications.
    points = points.drop(columns=["is_anomaly", "event_id", "label_status"], errors="ignore")
    points["context_label"] = points["context_label"].map(CONTEXT_LABELS).fillna("Calendário indisponível")
    eligible = points["eligible"].fillna(False).astype(bool)
    alerted = eligible & points["flag"].fillna(False).astype(bool)
    points["flag"] = alerted
    unknown_count = int((~eligible).sum())
    cards = st.columns(3)
    cards[0].metric("Dias analisados", str(int(eligible.sum())), border=True)
    cards[1].metric("Dias para investigar", str(int(alerted.sum())), border=True)
    cards[2].metric("Dias sem decisão", str(unknown_count), border=True)
    first, last = points["data"].iloc[0], points["data"].iloc[-1]
    st.caption(f"Período analisado: {first:%d/%m/%Y} a {last:%d/%m/%Y}. Os dias anteriores foram usados para construir a referência e definir os critérios de alerta.")
    if unknown_count:
        st.info(f"{unknown_count} dia(s) ficaram sem decisão por falta de informação suficiente. Eles não foram classificados como comportamento esperado.")
    st.altair_chart(context_comparison_chart(points), width="stretch")
    st.caption("Linha verde: o que foi vendido. Linha roxa tracejada: referência para o dia. Faixa roxa: faixa usada na comparação. Pontos vermelhos: dias sinalizados.")
    alerts = points.loc[alerted, ["data", "valor", "expected", "context_label"]].copy()
    st.markdown("**Dias para investigar**")
    if alerts.empty:
        st.info("Nenhum dos dias avaliados ultrapassou os critérios de alerta desta análise." if eligible.any()
                else "Nenhum dia pôde ser avaliado com as informações disponíveis.")
    else:
        alerts["data"] = alerts["data"].dt.strftime("%d/%m/%Y")
        alerts = alerts.rename(columns={"data": "Dia", "valor": "Vendas observadas",
                                         "expected": "Referência para o dia", "context_label": "Calendário informado"})
        st.dataframe(alerts.style.format({"Vendas observadas": _money, "Referência para o dia": _money}, na_rep="—"),
                     hide_index=True, width="stretch")
    _inspect_day(points, show_truth=False, prefer_alert=True)


def render_context_experiment():
    st.markdown("**Promoções e quedas inesperadas**")
    fixture = st.session_state.get("context_fixture_dir")
    st.caption("Comparação dos arquivos da base fictícia carregada e de seu calendário." if fixture
               else "Este experimento usa uma base fictícia própria, com 420 dias. Ele funciona localmente, sem API de IA.")
    st.info("Nesta simulação, promoções aumentam a quantidade vendida e aplicam desconto. O teste procura quedas extras, inclusive durante promoções.")
    if fixture:
        submitted = st.button("Recalcular comparação da base carregada", icon=":material/refresh:", key="recalculate_context_fixture")
    else:
        with st.form("context_experiment_settings"):
            text = st.text_input("Sementes para gerar as execuções", value="42,137,2026", key="context_experiment_seeds")
            submitted = st.form_submit_button("Gerar base e comparar", icon=":material/play_arrow:", type="primary")
    if submitted:
        try:
            with st.spinner("Comparando os dois métodos..."):
                if fixture:
                    result = evaluate_loaded_fixture(fixture)
                else:
                    seeds = _seeds(text)
                    result = run_contextual_benchmark(ROOT / "outputs/contextual" / "_".join(map(str, seeds)), seeds=seeds)
            st.session_state.context_experiment_result = result
        except ValueError as error:
            st.warning(str(error))
    result = st.session_state.get("context_experiment_result")
    if result is None:
        return
    st.caption("Execuções: " + ", ".join(map(str, result["seeds"])))
    _metrics(result)
    path = Path(result["outputs"]["point_results.csv"])
    all_points = _read_points(str(path), path.stat().st_mtime_ns)
    all_points["context_label"] = all_points["context_label"].map(CONTEXT_LABELS).fillna("Calendário indisponível")
    if st.session_state.get("context_result_seed") not in result["seeds"]:
        st.session_state.pop("context_result_seed", None)
    seed = st.selectbox("Execução para examinar", result["seeds"], key="context_result_seed")
    method_label = st.segmented_control("Referência exibida", list(METHODS.values()),
                                        default="Com calendário", required=True, key="context_result_method")
    method = next(key for key, value in METHODS.items() if value == method_label)
    points = all_points.loc[all_points["seed"].eq(seed) & all_points["detector"].eq(method)
                            & all_points["split"].eq("test")].copy().sort_values("data")
    if points.empty:
        st.warning("Este método não produziu pontos para examinar. Consulte os detalhes do experimento.")
    else:
        st.altair_chart(context_comparison_chart(points), width="stretch")
        st.caption("Verde: vendas observadas. Roxo tracejado: referência. Faixa: limites operacionais. Pontos vermelhos: alertas.")
        _inspect_day(points)
    with st.expander("Baixar os dados e os resultados"):
        labels = {"vendas.csv": "Vendas fictícias", "calendario.csv": "Calendário de promoções",
                  "gabarito.csv": "Gabarito separado", "point_results.csv": "Resultados por dia",
                  "metrics.csv": "Métricas por execução", "report.md": "Relatório do experimento"}
        for name, label in labels.items():
            if name in result["outputs"]:
                file = Path(result["outputs"][name])
                st.download_button("Baixar " + label.lower(), file.read_bytes, file_name=name,
                                   key="context_download_" + name, on_click="ignore")
    with st.expander("Procedimento e limitações"):
        st.markdown("Os dados são divididos por data: uma parte para aprender o padrão, outra para definir o limite de alerta e uma terceira para testar. O calendário é separado do gabarito.")
        st.markdown("O efeito das promoções foi definido no gerador. Este resultado ilustra o uso de contexto; a avaliação com dados reais e outras formas de mudança continua necessária.")
        st.json({"runs": result["runs"], "limitations": result.get("limitations", [])})
