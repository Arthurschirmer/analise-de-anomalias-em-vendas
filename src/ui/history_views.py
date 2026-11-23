"""Browse and reopen local snapshots without repeating an investigation."""

import pandas as pd
import streamlit as st

from src.evaluation.investigation_store import dataset_fingerprint, list_investigations, load_investigation
from src.formatting import format_currency


def _entry_label(entry):
    record = entry["record"]
    report = record["structured_report"]
    periods = report.get("periodos", {})
    created = pd.Timestamp(record["created_at"]).strftime("%d/%m/%Y %H:%M")
    window = report.get("contexto_da_base", {}).get("comparison_window", {})
    recut = f" · dias {window['first_day']}–{window['last_day']}" if window.get("mode") == "aligned_days" else ""
    return f"{created} · {entry['dataset_label']} · {periods.get('periodo_anterior', '—')} → {periods.get('periodo_atual', '—')}{recut} · {entry['id'][:6]}"


def render_investigation_history(data, directory, restore):
    st.subheader("Investigações salvas", icon=":material/history:")
    st.caption("As investigações mensais concluídas são guardadas neste computador. Reabrir um registro não executa ferramentas nem consulta a IA.")
    fingerprint = dataset_fingerprint(data)
    entries, invalid = list_investigations(directory)
    if invalid:
        st.warning(f"{invalid} registro(s) não puderam ser lidos. Os demais continuam disponíveis.")
    if not entries:
        st.info("As investigações entre meses aparecerão aqui depois de serem concluídas.")
        return
    active = [entry for entry in entries if entry["dataset_sha256"] == fingerprint]
    all_sources = st.toggle("Mostrar também registros de outras bases", key="history_all_sources")
    visible = entries if all_sources else active
    if not visible:
        st.caption("Há registros de outras bases. Carregue a mesma base ou ative a opção acima para conferir suas identificações.")
        return
    by_id = {entry["id"]: entry for entry in visible}
    if st.session_state.get("history_selected") not in by_id:
        st.session_state.pop("history_selected", None)
    selected = st.selectbox("Investigação salva", list(by_id), format_func=lambda key: _entry_label(by_id[key]), key="history_selected")
    entry = by_id[selected]
    matches = entry["dataset_sha256"] == fingerprint
    if not matches:
        st.caption("Este registro pertence a outra base ou a outras definições. Carregue a mesma base para reabri-lo.")
    if st.button("Reabrir investigação", icon=":material/restore:", disabled=not matches, key="history_restore"):
        try:
            restore(load_investigation(directory, selected, expected_dataset=fingerprint))
            st.rerun()
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"Não foi possível reabrir o registro: {error}")
    if len(active) < 2:
        return
    active_ids = {entry["id"]: entry for entry in active}
    old_selection = st.session_state.get("history_compare", [])
    if any(key not in active_ids for key in old_selection):
        st.session_state.pop("history_compare", None)
    chosen = st.multiselect("Comparar duas investigações desta base", list(active_ids),
        format_func=lambda key: _entry_label(active_ids[key]), max_selections=2, key="history_compare")
    if len(chosen) == 2:
        records = [active_ids[key]["record"]["structured_report"] for key in chosen]
        rows = []
        for key, label in (("faturamento_anterior", "Faturamento de referência"), ("faturamento_atual", "Faturamento em análise"),
                           ("variacao_faturamento", "Diferença de faturamento"), ("variacao_quantidade", "Diferença de quantidade")):
            row = {"Indicador": label}
            for index, report in enumerate(records, 1):
                value = report.get("resumo_comparativo", {}).get(key)
                row[f"Registro {index}"] = str(value) if key == "variacao_quantidade" else format_currency(value, report)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Confira os períodos e os recortes identificados em cada registro. Diferenças de configuração ou intervalo podem mudar os resultados; esta tabela não mede qual método é melhor.")
