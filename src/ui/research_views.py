"""Historical decision limits and transparent academic evaluation views."""
import hashlib
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.analysis.historical_detection import (
    apply_threshold, build_time_series, calibrate_threshold,
    score_historical_series, temporal_split,
)
from src.agent.investigation_agent import conclusion_request
from src.evaluation.anomaly_benchmark import clean_json, controlled_sales, run_anomaly_benchmark
from src.evaluation.llm_comparison import compare_investigation_reports, save_comparison
from src.evaluation.manual_review import JUDGEMENTS, read_reviews, review_scope, save_review
from src.ui.feedback import sanitize_details
from src.ui.review_views import render_sample_builder, render_review_panel

ROOT = Path(__file__).resolve().parents[2]


def historical_chart(frame):
    plot = frame.copy()
    plot["etapa"] = plot["split"].map({"train": "Treino", "calibration": "Calibração", "test": "Teste"})
    plot["alerta"] = plot["flag"] & plot["split"].eq("test")
    # Explicit segments avoid connecting lines across unknown observations.
    plot["segmento_observado"] = plot["valor"].isna().cumsum()
    plot["segmento_referencia"] = plot["expected"].isna().cumsum()
    base = alt.Chart(plot).encode(x=alt.X("data:T", title="Data"))
    tooltip = [alt.Tooltip("data:T", title="Data"), alt.Tooltip("etapa:N", title="Etapa"),
               alt.Tooltip("valor:Q", title="Observado", format=",.2f"),
               alt.Tooltip("expected:Q", title="Referência", format=",.2f"),
               alt.Tooltip("lower:Q", title="Limite inferior", format=",.2f"),
               alt.Tooltip("upper:Q", title="Limite superior", format=",.2f"),
               alt.Tooltip("n_history:Q", title="Pontos anteriores")]
    band = base.mark_area(opacity=.12, color="#6357D9").encode(
        y=alt.Y("lower:Q", title="Valor"), y2="upper:Q", detail="segmento_referencia:N")
    reference = base.mark_line(color="#6357D9", strokeDash=[6, 4]).encode(
        y="expected:Q", detail="segmento_referencia:N")
    actual = base.mark_line(color="#26A69A").encode(
        y="valor:Q", detail="segmento_observado:N", tooltip=tooltip)
    alerts = base.transform_filter(alt.datum.alerta).mark_point(
        color="#EF6B73", size=90, filled=True).encode(y="valor:Q", tooltip=tooltip)
    boundaries = plot.loc[plot["split"].ne(plot["split"].shift()), ["data", "etapa"]].iloc[1:]
    rules = alt.Chart(boundaries).mark_rule(color="#94A3B8", strokeDash=[3, 3]).encode(
        x="data:T", tooltip="etapa:N")
    return (band + reference + actual + alerts + rules).properties(height=350).interactive()


@st.cache_data(ttl=600, max_entries=6, show_spinner=False)
def calculate_history(data, configuration):
    series = build_time_series(data, frequency=configuration["frequency"],
                               metric=configuration["metric"], missing_policy=configuration["missing_policy"])
    weekly = configuration["frequency"] != "D"
    series = temporal_split(series, min_train=8 if weekly else 28,
                            min_calibration=5 if weekly else 7, min_test=5 if weekly else 7)
    scored = score_historical_series(series, window=configuration["window"],
                                     min_history=configuration["min_history"], trend=configuration["trend"])
    calibration = calibrate_threshold(scored, quantile=configuration["quantile"],
                                      method=configuration.get("calibration_method", "robust"),
                                      robust_multiplier=configuration.get("robust_multiplier", 3.0))
    return apply_threshold(scored, calibration), calibration


@st.cache_data(ttl=600, max_entries=6, show_spinner=False)
def calculate_segments(data, configuration):
    from src.analysis.segment_detection import detect_segment_anomalies
    return detect_segment_anomalies(
        data, dimension=configuration["dimension"], configuration=configuration,
        min_train_observations=8 if configuration["frequency"] != "D" else 28,
        max_segments=configuration["max_segments"], min_train_share=configuration["min_train_share"])


def _render_segment_result(data, result, config):
    summary, points, metadata = result["summary"], result["points"], result["metadata"]
    evaluated = summary.loc[summary["status"].eq("evaluated")]
    with st.container(horizontal=True):
        st.metric("Segmentos com cálculo", str(len(evaluated)), border=True)
        st.metric("Segmentos fora da avaliação", str(len(summary) - len(evaluated)), border=True)
        st.metric("Alertas nos segmentos", str(int(summary["alerts"].sum())), border=True)
    st.caption("Cada categoria ou região tem sua referência e seu limiar. A seleção usa somente o treino; pontos sem suporte suficiente ficam sem decisão.")
    reasons = {
        "": "Avaliado", "not_observed_in_training": "Sem observações no treino",
        "insufficient_training_observations": "Poucas observações no treino",
        "below_training_share": "Participação abaixo do mínimo",
        "outside_training_top_limit": "Fora do limite de segmentos",
        "insufficient_calibration_support": "Calibração insuficiente",
        "no_eligible_test_points": "Sem pontos avaliáveis no teste",
    }
    display = summary.sort_values(["selected", "alerts", "train_value"], ascending=False).copy()
    display["situacao"] = display["reason"].map(reasons).fillna(display["reason"])
    columns = ["segmento", "train_share", "train_observations", "eligible_test_points",
               "alerts", "unknown_test_points", "threshold", "situacao"]
    st.dataframe(display[columns].rename(columns={
        "segmento": "Segmento", "train_share": "Participação no treino",
        "train_observations": "Períodos observados no treino", "eligible_test_points": "Pontos avaliáveis no teste",
        "alerts": "Alertas", "unknown_test_points": "Pontos incompletos no teste",
        "threshold": "Limiar", "situacao": "Situação"}), hide_index=True)
    with st.expander("Critérios e limites da análise por segmento"):
        st.json({"configuration": result["configuration"], "splits": metadata["split_boundaries"]})
        for item in metadata["limitations"]:
            st.caption(item)
    if points.empty:
        st.info("Nenhum segmento teve dados suficientes para calcular a referência e calibrar o limiar.")
        return
    st.download_button("Baixar resultados por segmento", points.to_csv(index=False).encode("utf-8-sig"),
                       "anomalias_segmentos.csv", mime="text/csv", key="download_segments")
    points.attrs["split_boundaries"] = metadata["split_boundaries"]
    render_sample_builder(data, points, result["configuration"], ROOT / "outputs/reviews")
    options = evaluated["segmento"].astype(str).tolist()
    if st.session_state.get("historical_segment") not in options:
        st.session_state.pop("historical_segment", None)
    selected = st.selectbox("Segmento para explorar", options, key="historical_segment")
    scored = points.loc[points["segmento"].eq(selected)].copy()
    scored.attrs["split_boundaries"] = metadata["split_boundaries"]
    config = {**config, "segmento": selected}
    _render_series_result(data, scored, metadata["calibrations"][selected], config, allow_sample=False)


def render_history(data, load_demo):
    st.subheader("O que foge do histórico?", icon=":material/timeline:")
    st.caption("Compare o valor observado com uma referência calculada apenas com dados anteriores.")
    if (data["data"].max() - data["data"].min()).days < 56 or data["data"].nunique() < 56:
        st.info("Esta base é curta ou esparsa para separar treino, calibração e teste. Carregue um histórico maior ou experimente a demonstração de 420 dias.")
        if st.button("Carregar demonstração histórica", icon=":material/science:", key="load_historical_demo"):
            load_demo(controlled_sales())
            st.rerun()
    with st.form("historical_detection", border=True):
        left, right = st.columns(2)
        scope = left.selectbox("Analisar", ["Base inteira", "Por categoria", "Por região / UF"], key="historical_scope")
        calibration_rule = right.selectbox("Regra de calibração", ["Mediana e MAD dos escores", "Quantil dos escores"], key="historical_calibration")
        frequency = left.selectbox("Agregação", ["Diária", "Semanal"], key="historical_frequency")
        metric = right.selectbox("Métrica histórica", ["Faturamento", "Quantidade"], key="historical_metric")
        window = left.number_input("Janela de referência", min_value=4, max_value=26, value=8, step=1)
        minimum = right.number_input("Observações mínimas anteriores", min_value=3, max_value=26, value=4, step=1)
        trend = left.checkbox("Considerar tendência local", value=True)
        missing = right.selectbox("Datas sem registros", ["Manter como desconhecidas", "Assumir vendas iguais a zero"])
        quantile = st.slider("Quantil de calibração (%)", 90, 100, 98)
        with st.expander("Filtros para análise de segmentos"):
            st.caption("A escolha dos segmentos usa somente o treino. Grupos sem suporte suficiente ficam identificados como não avaliados.")
            max_segments = st.number_input("Máximo de segmentos", min_value=1, max_value=30, value=12, step=1)
            min_share = st.number_input("Participação mínima no treino (%)", min_value=0.0, max_value=100.0, value=1.0, step=.5)
        st.caption("Diária: mesmo dia da semana em semanas anteriores. Semanal: semanas completas anteriores. Treino 50% · calibração 25% · teste restante.")
        submitted = st.form_submit_button("Calcular referência histórica", type="primary", icon=":material/query_stats:")
    if submitted:
        config = {"frequency": "D" if frequency == "Diária" else "W-SUN",
                  "metric": "faturamento" if metric == "Faturamento" else "quantidade",
                  "window": int(window), "min_history": int(minimum),
                  "trend": "linear" if trend else "none", "quantile": quantile / 100,
                  "calibration_method": "robust" if calibration_rule.startswith("Mediana") else "quantile",
                  "robust_multiplier": 3.0,
                  "missing_policy": "unknown" if missing.startswith("Manter") else "zero"}
        try:
            with st.spinner("Calculando com referências anteriores a cada observação..."):
                if scope == "Base inteira":
                    scored, calibration = calculate_history(data, config)
                    st.session_state.historical_result = (scored, calibration, config)
                    st.session_state.pop("segment_result", None)
                else:
                    dimension = "categoria" if scope == "Por categoria" else "regiao"
                    config.update(dimension=dimension, max_segments=int(max_segments), min_train_share=min_share / 100)
                    result = calculate_segments(data, config)
                    st.session_state.segment_result = (result, config)
                    st.session_state.pop("historical_result", None)
            st.session_state.pop("historical_alert_date", None)
            st.session_state.pop("historical_segment", None)
        except ValueError as error:
            st.warning(str(error))
            st.info("Se houver um resultado abaixo, ele corresponde ao último cálculo válido; o novo cálculo não foi aplicado.")
    segment_result = st.session_state.get("segment_result")
    if submitted and (st.session_state.get("historical_result") is not None or segment_result is not None):
        st.caption("A configuração aplicada está identificada junto ao resultado; alterar os controles exige executar o cálculo novamente.")
    if segment_result is not None:
        _render_segment_result(data, *segment_result)
        return
    result = st.session_state.get("historical_result")
    if result is None:
        st.caption("Os limites e alertas aparecem depois do cálculo. Nenhuma chamada à IA é necessária.")
        return
    scored, calibration, config = result
    _render_series_result(data, scored, calibration, config)


def _render_series_result(data, scored, calibration, config, allow_sample=True):
    st.caption("Resultado calculado: " + ("diário" if config["frequency"] == "D" else "semanal") +
               f' · janela {config["window"]} · mínimo {config["min_history"]} · calibração ' +
               ("mediana e MAD" if config.get("calibration_method", "robust") == "robust" else "quantil"))
    test = scored.loc[scored["split"].eq("test")]
    alerts = test.loc[test["eligible"] & test["flag"]]
    with st.container(horizontal=True):
        st.metric("Pontos avaliáveis no teste", str(int(test["eligible"].sum())), border=True)
        st.metric("Alertas para revisar", str(len(alerts)), border=True)
        st.metric("Limiar congelado", f'{calibration["threshold"]:.2f}', border=True)
    with st.container(border=True):
        st.markdown("**Observado, referência e limites de decisão**")
        st.altair_chart(historical_chart(scored), width="stretch")
        st.caption("Verde: observado · roxo tracejado: referência · faixa roxa: limites de decisão · vermelho: alertas no teste. Linhas verticais separam treino, calibração e teste.")
        st.caption("Os limites são regras empíricas de decisão, sem interpretação de intervalo de confiança. A referência se atualiza com o passado; o limiar é fixado na calibração.")
    with st.expander("Como o resultado foi calculado"):
        st.json({"configuration": config, "calibration": calibration, "splits": scored.attrs["split_boundaries"]})
        st.markdown("A tendência usa a mediana das inclinações entre pares anteriores. A escala usa MAD dos resíduos × 1,4826, com piso de 5% da referência ou 1 unidade. Ela não modela feriados nem sazonalidade anual.")
        st.caption("Desconhecidos são excluídos. Semanas parciais ficam inelegíveis. Status finais da Olist tornam a análise retrospectiva.")
        st.caption("A calibração por mediana e MAD reduz a influência de poucos escores extremos; ainda é uma regra empírica e pode aumentar a quantidade de alertas.")
    st.download_button("Baixar série e limites", scored.to_csv(index=False).encode("utf-8-sig"),
                       "serie_historica.csv", mime="text/csv", key="download_historical_series")
    if allow_sample:
        render_sample_builder(data, scored, config, ROOT / "outputs/reviews")
    if alerts.empty:
        st.info("Nenhum alerta no teste com esta configuração. Isso não comprova ausência de anomalias.")
        return
    st.dataframe(alerts[["periodo", "valor", "expected", "lower", "upper", "score"]].rename(columns={
        "periodo": "Período", "valor": "Observado", "expected": "Referência", "lower": "Limite inferior",
        "upper": "Limite superior", "score": "Escore"}), hide_index=True)
    _render_review(data, scored, alerts, config)


def _render_review(data, scored, alerts, config):
    st.markdown("**Revisão humana dos alertas**")
    st.caption("O detector sinaliza mudanças; o parecer deve considerar o contexto comercial e a qualidade da fonte.")
    periods = alerts["periodo"].tolist()
    if st.session_state.get("historical_alert_date") not in periods:
        st.session_state.pop("historical_alert_date", None)
    period = st.selectbox("Alerta para revisar", periods, key="historical_alert_date")
    selected = scored.loc[scored["periodo"].eq(period)].iloc[0]
    observed = data.loc[data["data"].dt.normalize().between(selected["inicio_periodo"], selected["fim_periodo"])]
    if config.get("dimension") and config.get("segmento"):
        labels = observed[config["dimension"]].astype("string").fillna("(não informado)").astype(str)
        observed = observed.loc[labels.eq(config["segmento"])]
    if not observed.empty:
        frame = observed.groupby("categoria", as_index=False)["faturamento"].sum()
        st.bar_chart(frame.nlargest(10, "faturamento"), x="categoria", y="faturamento",
                     x_label="Categoria", y_label="Faturamento observado")
        st.caption("Distribuição observada no período selecionado; o gráfico não atribui uma causa ao desvio.")
    scope = review_scope(data, config)
    reviews = read_reviews(scope, ROOT / "outputs/reviews")
    current = reviews.get(period, {})
    if current:
        st.caption("Parecer salvo: " + current["judgement"] + " · " + current["reviewer"])
    widget_scope = scope[:12] + "_" + period
    with st.form("human_alert_review"):
        judgement = st.selectbox("Parecer", JUDGEMENTS, key="alert_judgement_" + widget_scope,
                                index=JUDGEMENTS.index(current["judgement"]) if current else 0)
        reviewer = st.text_input("Responsável pela revisão")
        notes = st.text_area("Evidência ou justificativa", value=current.get("notes", ""), key="alert_notes_" + widget_scope)
        if st.form_submit_button("Salvar revisão"):
            try:
                reviews = save_review(scope, period, judgement, reviewer, notes, ROOT / "outputs/reviews")
                st.success("Revisão salva para esta base e configuração.")
            except ValueError as error:
                st.warning(str(error))
    visible = [item for key, item in reviews.items() if key in set(alerts["periodo"])]
    st.caption(f"{len(visible)} de {len(alerts)} alertas revisados. Revisar só alertas não permite medir recall global.")
    if visible:
        st.dataframe(pd.DataFrame(visible), hide_index=True)
        st.download_button("Baixar pareceres", pd.DataFrame(visible).to_csv(index=False).encode("utf-8-sig"),
                           "revisoes.csv", mime="text/csv")


def render_evaluation(data, investigation, provider_factory):
    st.subheader("Avaliação do trabalho", icon=":material/science:")
    mode = st.selectbox("Experimento", ["Detecção de anomalias", "Relatório local e IA", "Revisão humana"], key="evaluation_mode")
    if mode == "Revisão humana":
        render_review_panel()
        return
    if mode == "Relatório local e IA":
        _render_explanation_comparison(investigation, provider_factory)
        return
    st.markdown("**Um experimento reproduzível com respostas conhecidas**")
    st.caption("420 dias fictícios com tendência, padrão semanal e ruído. Compare o baseline, o quantil anterior e a nova calibração por mediana e MAD, usando os mesmos pontos no teste.")
    olist = data.attrs.get("dataset_context", {}).get("source_type") == "olist"
    if olist:
        st.info("A base Olist carregada também será copiada para receber perturbações artificiais. Os arquivos originais são preservados; pontos reais sem injeção continuam sem rótulo.")
    with st.form("benchmark_settings"):
        seed = st.number_input("Semente do experimento", 0, 2**32 - 1, 42, step=1)
        run = st.form_submit_button("Executar avaliação local", type="primary", icon=":material/play_arrow:")
    if run:
        try:
            with st.spinner("Executando os detectores sem usar a API..."):
                result = run_anomaly_benchmark(
                    ROOT / "outputs/benchmark" / review_scope(data, {"seed": int(seed)})[:16],
                    seed=int(seed), real_data=data if olist else None)
            st.session_state.benchmark_result = result
        except ValueError as error:
            st.warning(str(error))
    result = st.session_state.get("benchmark_result")
    _render_benchmark_suite(data, olist)
    if result is None:
        return
    for stage_name, stage in result["stages"].items():
        with st.container(border=True):
            st.markdown("**Dados controlados**" if stage_name == "controlled" else "**Olist com injeções artificiais**")
            st.caption(stage["label_definition"])
            frame = pd.DataFrame(stage["metrics"])
            if stage["ground_truth_complete"]:
                columns = ["detector", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
            else:
                columns = ["detector", "injection_recall", "known_positive_points", "tp", "fn", "unlabeled_alerts"]
                st.caption("Precisão e F1 reais não são calculados. Alertas sem rótulo precisam de revisão humana.")
                for item in stage["metrics"]:
                    if item["known_positive_points"] and item["tp"] == 0:
                        st.warning(f'{item["detector"]}: nenhuma das injeções avaliáveis foi detectada neste experimento. Examine o limiar e a variabilidade da calibração; não ajuste usando os rótulos do teste para depois tratá-lo como validação independente.')
            st.dataframe(frame[columns].rename(columns={
                "detector": "Detector", "precision": "Precisão", "recall": "Recall", "f1": "F1",
                "injection_recall": "Recall das injeções", "known_positive_points": "Pontos injetados avaliáveis",
                "unlabeled_alerts": "Alertas sem rótulo", "tp": "VP", "fp": "FP", "fn": "FN", "tn": "VN"}),
                hide_index=True)
            with st.expander("Eventos, partições e calibração"):
                st.dataframe(pd.DataFrame(stage["events"]), hide_index=True)
                st.json({"splits": stage["splits"], "calibration": stage["calibration"],
                         "input_sha256": stage["input_sha256"]})
    st.caption(" · ".join(result["limitations"]))
    for name, path in result["outputs"].items():
        file = Path(path)
        if file.exists():
            st.download_button("Baixar " + name, file.read_bytes, file_name=name,
                               key="benchmark_download_" + name, on_click="ignore")


def _render_benchmark_suite(data, is_olist):
    st.markdown("**Robustez em diferentes cenários e sementes**")
    st.caption("Quedas e picos de 10%, 30% e 60%, queda gradual, mudança persistente e problemas em categoria/UF compensados no faturamento total.")
    with st.form("benchmark_suite_settings"):
        seeds_text = st.text_input("Sementes separadas por vírgula", value="42,137,2026", key="benchmark_suite_seeds")
        submitted = st.form_submit_button("Executar suíte ampliada", icon=":material/science:")
    if submitted:
        try:
            from src.evaluation.benchmark_suite import run_benchmark_suite
            seeds = tuple(int(part.strip()) for part in seeds_text.split(","))
            if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds):
                raise ValueError("Informe de 1 a 5 sementes distintas para executar no site.")
            identity = review_scope(data, {"seeds": seeds, "protocol": "suite_v1"})
            with st.spinner("Avaliando os cenários e preservando os resultados de cada semente..."):
                result = run_benchmark_suite(
                    ROOT / "outputs/benchmark/suite" / identity[:16],
                    seeds=seeds, real_data=data if is_olist else None)
            st.session_state.benchmark_suite_result = result
        except ValueError as error:
            st.warning(str(error))
    result = st.session_state.get("benchmark_suite_result")
    if result is None:
        return
    st.caption("Sementes usadas: " + ", ".join(map(str, result["seeds"])))
    frame = pd.DataFrame(result["aggregate_metrics"])
    if frame.empty:
        st.info("A suíte não produziu pontos suficientes para comparar os métodos.")
        return
    stage = st.selectbox("Dados da suíte", frame["stage"].unique().tolist(),
                         format_func=lambda key: "Dados controlados" if key == "controlled" else "Olist com injeções")
    subset = frame.loc[frame["stage"].eq(stage)]
    if st.session_state.get("suite_scope") not in subset["scope"].unique().tolist():
        st.session_state.pop("suite_scope", None)
    scope = st.selectbox("Série avaliada na suíte", subset["scope"].unique().tolist(), key="suite_scope")
    subset = subset.loc[subset["scope"].eq(scope)].copy()
    labels = {"previous_week_30pct": "Variação semanal 30%", "historical_quantile98": "Histórico · quantil 98%",
              "seasonal_trend_mad": "Histórico · mediana e MAD"}
    subset["metodo"] = subset["detector"].map(labels)
    if stage == "controlled":
        metric = "f1_mean"
        columns = ["scenario", "metodo", "runs", "precision_mean", "recall_mean", "f1_mean", "f1_std", "f1_n", "false_alert_rate_mean"]
    else:
        metric = "injection_recall_mean"
        columns = ["scenario", "metodo", "runs", "injection_recall_mean", "injection_recall_std", "injection_recall_n", "unlabeled_alerts_mean"]
        st.caption("Recall das injeções é parcial. Não há precisão/F1 reais sem rótulos dos demais pontos.")
    st.dataframe(subset[[col for col in columns if col in subset]].rename(columns={
        "scenario": "Cenário", "metodo": "Método", "runs": "Execuções",
        "precision_mean": "Precisão média", "recall_mean": "Recall médio", "f1_mean": "F1 médio",
        "f1_std": "Desvio do F1", "fp_mean": "FP médios", "fn_mean": "FN médios",
        "f1_n": "Execuções com F1 válido", "false_alert_rate_mean": "Taxa média de falsos alertas",
        "injection_recall_n": "Execuções com recall válido",
        "injection_recall_mean": "Recall médio das injeções", "injection_recall_std": "Desvio do recall das injeções",
        "unlabeled_alerts_mean": "Alertas médios sem rótulo"}), hide_index=True)
    st.caption("As médias usam somente valores definidos. O número de execuções com métrica válida está indicado. Em controles sem positivos, F1 fica em branco; examine a taxa de falsos alertas nos dados controlados.")
    plot = subset.loc[subset[metric].notna()]
    if not plot.empty:
        chart = alt.Chart(plot).mark_bar().encode(
            x=alt.X("scenario:N", title="Cenário", sort=None), xOffset="metodo:N",
            y=alt.Y(metric + ":Q", title="F1 médio" if stage == "controlled" else "Recall médio das injeções",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metodo:N", title="Método"),
            tooltip=["scenario:N", "metodo:N", alt.Tooltip(metric + ":Q", format=".3f"), "runs:Q"],
        ).properties(height=290)
        st.altair_chart(chart, width="stretch")
    st.caption("Médias e desvios entre sementes descrevem estas execuções. Não representam intervalos de confiança. Na Olist, as sementes mudam a posição das injeções sobre o mesmo histórico.")
    with st.expander("Eventos, atraso de detecção e protocolo"):
        st.dataframe(pd.DataFrame(result["event_metrics"]), hide_index=True)
        st.json({"configuration": result["configuration"], "manifests": result["manifests"]})
        for limitation in result["limitations"]:
            st.caption(limitation)
    for name, path in result["outputs"].items():
        file = Path(path)
        if file.exists():
            st.download_button("Baixar suíte · " + name, file.read_bytes, file_name=name,
                               key="suite_download_" + name, on_click="ignore")


def _render_explanation_comparison(investigation, provider_factory):
    st.markdown("**Mesmas evidências, duas explicações**")
    st.caption("A resposta local e a IA recebem o mesmo contexto e os mesmos prompts. A avaliação de clareza, fidelidade e utilidade é humana.")
    if investigation is None:
        st.info("Faça uma investigação de dois meses antes de comparar os relatórios.")
        return
    include_ai = st.checkbox("Consultar a IA configurada nesta comparação", value=False,
                             help="Envia as evidências da investigação à API configurada.")
    if st.button("Comparar explicações", type="primary", key="compare_explanations"):
        try:
            provider = provider_factory() if include_ai else None
            with st.spinner("Preparando a comparação..."):
                result = compare_investigation_reports(investigation, provider)
            st.session_state.explanation_comparison = result
            for key in list(st.session_state):
                if key.startswith("grade_") or key == "explanation_review_notes":
                    st.session_state.pop(key, None)
            save_comparison(result, ROOT / "outputs/model_comparison")
        except Exception as error:
            st.warning(sanitize_details(str(error)))
    result = st.session_state.get("explanation_comparison")
    if result is None:
        return
    _, _, current_context = conclusion_request(
            investigation.anomaly_description, investigation.evidence, investigation.structured_report)
    digest = hashlib.sha256(json.dumps(clean_json(current_context), ensure_ascii=False,
                                      sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    if result["evidence_sha256"] != digest:
        st.info("A investigação mudou. Execute novamente a comparação para usar as evidências atuais.")
        return
    for col, name in zip(st.columns(2), ["local", "ai"]):
        entry = result[name]
        with col, st.container(border=True):
            st.markdown("**Análise local**" if name == "local" else "**Explicação da IA**")
            if entry["status"] == "completed":
                st.markdown(entry["text"])
                st.caption(f'Tempo medido: {entry["latency_seconds"]:.3f} s')
                check = entry["numeric_check"]
                st.caption(f'Números com correspondência nas evidências: {check["matched"]}/{check["total"]}. Verificação heurística.')
                if check["needs_review"]:
                    st.caption("Conferir no contexto: " + ", ".join(check["needs_review"]))
            elif entry["status"] == "failed":
                st.warning("A chamada à IA falhou: " + entry.get("error", ""))
            else:
                st.info("IA não consultada nesta execução.")
    completed = [key for key in ["local", "ai"] if result[key]["status"] == "completed"]
    with st.form("human_explanation_review"):
        st.markdown("**Parecer humano · escala de 1 a 5**")
        st.caption("1 indica desempenho insuficiente; 5 indica resposta clara, fiel e útil. Sem preenchimento, o parecer continua pendente.")
        reviewer = st.text_input("Responsável pelo parecer", key="explanation_reviewer")
        grades = {}
        for key in completed:
            st.markdown("**" + ("Local" if key == "local" else "IA") + "**")
            grades[key] = {rubric: st.selectbox(label, [None, 1, 2, 3, 4, 5],
                format_func=lambda value: "Não avaliado" if value is None else str(value),
                help=result["rubric"][rubric], key=f"grade_{key}_{rubric}")
                for rubric, label in [("fidelity", "Fidelidade"), ("clarity", "Clareza"), ("usefulness", "Utilidade")]}
        notes = st.text_area("Justificativa do parecer", key="explanation_review_notes")
        if st.form_submit_button("Salvar parecer das explicações"):
            if not reviewer.strip() or any(value is None for grade in grades.values() for value in grade.values()):
                st.warning("Preencha o responsável e todas as notas dos relatórios disponíveis.")
            else:
                result["human_review"] = {"status": "completed", "reviewer": reviewer.strip(),
                                          **grades, "notes": notes}
                save_comparison(result, ROOT / "outputs/model_comparison")
                st.success("Parecer salvo.")
    st.caption("Estado do parecer humano: " + result["human_review"]["status"])
    st.download_button("Baixar comparação completa",
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        file_name="comparacao_explicacoes.json", mime="application/json")


def render_revenue_explanation(report):
    decomposition = report.get("decomposicao_faturamento", {})
    if decomposition.get("available"):
        with st.container(border=True):
            st.markdown("**De onde vem a mudança de faturamento?**")
            if not decomposition["common_basket_available"]:
                st.info("Não há produtos comuns nos dois meses. Apenas entradas e saídas observadas podem ser decompostas neste recorte.")
            components = pd.DataFrame(decomposition["components"])
            chart = alt.Chart(components).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("value:Q", title="Contribuição para a diferença de faturamento (R$)"),
                y=alt.Y("label:N", sort=None, title=None),
                color=alt.condition(alt.datum.value >= 0, alt.value("#26A69A"), alt.value("#EF6B73")),
                tooltip=[alt.Tooltip("label:N", title="Componente"),
                         alt.Tooltip("value:Q", title="Contribuição (R$)", format=",.2f")],
            ).properties(height=220)
            st.altair_chart(chart, width="stretch")
            st.caption("A soma dos componentes reconcilia a diferença total. Volume e mix usam somente os produtos comuns aos dois meses; entradas e saídas são separadas.")
            with st.expander("Convenção e limites da decomposição"):
                st.markdown(decomposition["convention"])
                for item in decomposition["limitations"]:
                    st.caption(item)
                st.caption(f'Erro de reconciliação: {decomposition["reconciliation_error"]:.8f}')
                if decomposition["product_changes"]:
                    st.dataframe(pd.DataFrame(decomposition["product_changes"]), hide_index=True)
                    if decomposition["product_changes_truncated"]:
                        st.caption("Tabela dos maiores impactos; os cálculos usam todos os produtos.")
    elif decomposition:
        st.caption("Decomposição indisponível para este recorte: " + str(decomposition.get("reason")))
    operations = report.get("indicadores_operacionais", {})
    if operations.get("available"):
        with st.expander("Pedidos cancelados e entregas em atraso", icon=":material/local_shipping:"):
            st.dataframe(pd.DataFrame(operations["monthly"]).rename(columns={
                "periodo": "Mês da compra", "pedidos": "Pedidos", "cancelados": "Cancelados",
                "taxa_cancelamento": "Cancelamento (%)", "entregas_avaliaveis": "Entregas avaliáveis",
                "entregas_atrasadas": "Atrasadas", "taxa_atraso_observada": "Atraso observado (%)",
                "pedidos_fora_denominador_atraso": "Fora do denominador de atraso"}), hide_index=True)
            for item in operations["limitations"]:
                st.caption(item)

