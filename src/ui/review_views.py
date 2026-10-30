"""Human review panels with sampled alerts and non-alerts."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluation.manual_review import (
    JUDGEMENTS, create_review_sample, read_reviews, review_scope, save_review,
    save_review_sample, read_review_sample, summarize_review_sample,
)

def render_sample_builder(data, scored, configuration, output_dir):
    if not (scored["split"].eq("test") & scored["eligible"]).any():
        st.info("Não há pontos elegíveis no teste para preparar a revisão humana.")
        return
    with st.expander("Preparar avaliação humana com pontos sem alerta", icon=":material/fact_check:"):
        st.caption("A amostra sorteia alertas e pontos não sinalizados. Ela permite procurar também anomalias que o detector deixou passar.")
        with st.form("review_sample_settings"):
            size = st.number_input("Tamanho desejado da amostra", min_value=2, max_value=300,
                                   value=30, step=1, key="review_sample_size")
            seed = st.number_input("Semente da amostragem", min_value=0, max_value=2**32 - 1,
                                   value=42, step=1, key="review_sample_seed")
            blind = st.checkbox("Ocultar decisão do detector na ficha de revisão", value=True,
                                key="review_sample_blind")
            submitted = st.form_submit_button("Preparar amostra para revisão", type="primary")
        if submitted:
            try:
                sample = create_review_sample(scored, sample_size=int(size), seed=int(seed), blind=blind)
                context_columns = [col for col in ["data", "valor", "valor_observado", "observado",
                    "dias_observados", "dias_esperados", "motivo", "segmento", "dimensao"] if col in scored]
                sample.attrs["context_records"] = json.loads(scored[context_columns].to_json(orient="records", date_format="iso"))
                sample.attrs["analysis_boundaries"] = scored.attrs.get("split_boundaries", {})
                scope = review_scope(data, {"analysis": configuration, "review_protocol": "stratified_v1",
                                           "sample_id": sample.attrs["sample_id"]})
                existing = read_review_sample(scope, output_dir)
                if existing is not None:
                    sample = existing
                else:
                    save_review_sample(scope, sample, output_dir)
                st.session_state.review_panel = {
                    "scope": scope, "sample": sample, "configuration": configuration,
                    "output_dir": str(output_dir),
                }
                st.session_state.pop("review_case_id", None)
                st.success("Amostra salva. Abra Avaliação → Revisão humana para preencher os pareceres.")
            except ValueError as error:
                st.warning(str(error))


def render_review_panel():
    st.markdown("**Avaliação humana dos pontos selecionados**")
    if st.session_state.pop("review_saved_notice", False):
        st.success("Parecer salvo.")
    panel = st.session_state.get("review_panel")
    if panel is None:
        st.info("Calcule o Histórico e use “Preparar avaliação humana com pontos sem alerta” para gerar uma amostra.")
        return
    sample = panel["sample"]
    scope, output_dir = panel["scope"], Path(panel["output_dir"])
    reviews = read_reviews(scope, output_dir)
    summary = summarize_review_sample(sample, reviews)
    blind = bool(sample.attrs.get("blind", True))
    st.caption("O painel pertence à base e à configuração em que foi criado. O estado da revisão fica salvo em disco.")
    with st.expander("Protocolo e configuração desta amostra"):
        st.json({"configuration": panel["configuration"], "review_scope": scope,
                 "sample_id": sample.attrs["sample_id"], "seed": sample.attrs.get("seed"),
                 "splits": sample.attrs.get("analysis_boundaries", {}),
                 "decision_hidden_requested": blind})
    with st.container(horizontal=True):
        st.metric("Pontos na amostra", str(summary["sampled"]), border=True)
        st.metric("Pareceres preenchidos", str(summary["reviewed"]), border=True)
        st.metric("Inconclusivos", str(summary["inconclusive"]), border=True)
    if blind and summary["status"] != "complete":
        st.info("A decisão, os escores e os limites estão ocultos nesta ficha. O histórico continua acessível: esta opção, sozinha, não garante um estudo cego independente.")
    else:
        st.caption("Avalie a qualidade da fonte e o contexto de negócio antes de confirmar uma anomalia.")
    ids = sample["review_id"].astype(str).tolist()
    records = sample.set_index("review_id")

    def label(review_id):
        row = records.loc[review_id]
        group = str(row.get("segmento", ""))
        completed = " · revisado" if review_id in reviews else ""
        return str(row["periodo"]) + (" · " + group if group else "") + completed

    if st.session_state.get("review_case_id") not in ids:
        st.session_state.pop("review_case_id", None)
    review_id = st.selectbox("Ponto para revisar", ids, format_func=label, key="review_case_id")
    selected = records.loc[review_id]
    current = reviews.get(review_id)
    assumed = (not bool(selected.get("observado", True)) or
               int(selected.get("dias_observados", 1)) < int(selected.get("dias_esperados", 1)))
    st.metric("Valor analisado" if assumed else "Valor observado", f'{float(selected["valor"]):,.2f}')
    if assumed:
        st.info("Este valor inclui preenchimento de dias sem registros como zero. Isso é uma hipótese da análise e não comprova vendas observadas iguais a zero.")
    if current:
        st.caption("Parecer salvo: " + current["judgement"] + " · " + current["reviewer"])
    # Historical observations provide context without revealing the detector.
    context = sample.attrs.get("context_records", [])
    if context:
        history = pd.DataFrame(context)
        if "segmento" in history and "segmento" in selected:
            history = history.loc[history["segmento"].eq(selected["segmento"])]
        date = pd.Timestamp(selected["data"])
        history["data"] = pd.to_datetime(history["data"])
        history = history.loc[history["data"].between(date - pd.Timedelta(days=56), date)]
        if not history.empty:
            history = history.sort_values("data")
            history["line_segment"] = history["valor"].isna().cumsum()
            if "observado" in history:
                filled = ~history["observado"].fillna(False)
                if "dias_observados" in history:
                    filled |= history["dias_observados"].lt(history["dias_esperados"])
                history["tipo_valor"] = filled.map({True: "Inclui zero assumido", False: "Observado"})
            else:
                history["tipo_valor"] = "Valor analisado"
            chart = alt.Chart(history).mark_line(point=True, color="#26A69A").encode(
                x=alt.X("data:T", title="Data"), y=alt.Y("valor:Q", title="Valor analisado"),
                detail="line_segment:N",
                color=alt.Color("tipo_valor:N", title="Origem do valor"),
                tooltip=["data:T", "tipo_valor:N", alt.Tooltip("valor:Q", format=",.2f")]).properties(height=220)
            st.altair_chart(chart)
            st.caption("Valores até o ponto em revisão. Preenchimentos são identificados separadamente das observações.")
    if not blind or summary["status"] == "complete":
        st.caption("Decisão do detector: " + ("Alerta" if bool(selected["flag"]) else "Sem alerta"))
        st.caption(f'Referência: {selected.get("expected", float("nan")):,.2f}')
    judgement_key = f"sample_judgement_{scope[:12]}_{review_id}"
    notes_key = f"sample_notes_{scope[:12]}_{review_id}"
    if judgement_key not in st.session_state:
        st.session_state[judgement_key] = current["judgement"] if current else None
    if notes_key not in st.session_state:
        st.session_state[notes_key] = current.get("notes", "") if current else ""
    with st.form("sample_case_review"):
        judgement = st.selectbox("Parecer sobre o ponto", [None, *JUDGEMENTS],
                                format_func=lambda value: "Não avaliado" if value is None else value,
                                key=judgement_key)
        reviewer = st.text_input("Responsável pela avaliação", key="sample_review_reviewer")
        notes = st.text_area("Justificativa e evidências do parecer", key=notes_key)
        if st.form_submit_button("Salvar parecer do ponto", type="primary"):
            if judgement is None or not notes.strip():
                st.warning("Escolha um parecer e descreva a justificativa.")
            else:
                try:
                    reviews = save_review(scope, review_id, judgement, reviewer, notes, output_dir,
                        metadata={"periodo": str(selected["periodo"]),
                                  "segmento": str(selected.get("segmento", "")),
                                  "dimensao": str(selected.get("dimensao", "")),
                                  "sample_id": sample.attrs["sample_id"],
                                  "decision_hidden": blind and summary["status"] != "complete"})
                    st.session_state.review_saved_notice = True
                    st.rerun()
                except ValueError as error:
                    st.warning(str(error))
    visible = [col for col in ["review_id", "periodo", "segmento", "dimensao", "valor",
        "valor_observado", "observado", "dias_observados", "dias_esperados", "motivo"] if col in sample]
    export = sample[visible].copy()
    export["parecer"] = export["review_id"].map(lambda key: reviews.get(key, {}).get("judgement", ""))
    export["responsavel"] = export["review_id"].map(lambda key: reviews.get(key, {}).get("reviewer", ""))
    export["justificativa"] = export["review_id"].map(lambda key: reviews.get(key, {}).get("notes", ""))
    st.download_button("Baixar fichas e pareceres", export.to_csv(index=False).encode("utf-8-sig"),
                       "revisao_amostra.csv", mime="text/csv", key="download_review_panel")
    if summary["metrics"] is None:
        st.info("As métricas ficam pendentes até todos os pontos da amostra terem um parecer conclusivo.")
        st.caption(f'Estado: {summary["status"]} · faltantes: {summary["missing"]} · inconclusivos: {summary["inconclusive"]}')
    else:
        metrics = summary["metrics"]
        st.markdown("**Concordância do detector com os pareceres**")
        frame = pd.DataFrame([{
            "Precisão": metrics.get("precision"), "Recall": metrics.get("recall"),
            "F1": metrics.get("f1"), "Acurácia": metrics.get("accuracy"),
        }])
        st.dataframe(frame, hide_index=True)
        st.caption("Estimativas ponderadas pelo desenho da amostragem." if metrics["is_estimate"] else
                   "Resultados do censo dos pontos elegíveis deste teste.")
        st.caption("Os pareceres são a referência humana adotada, sujeitos a divergência e revisão. Não comprovam uma causa.")
        st.json(summary["coverage_by_stratum"])
        st.download_button("Baixar resumo da avaliação humana",
                           json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                           "avaliacao_humana.json", mime="application/json")
    for limitation in summary.get("limitations", []):
        st.caption(limitation)

