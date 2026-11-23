"""Guided upload, with preview, explicit mapping and structural validation."""

from hashlib import sha256

import streamlit as st

from src.data.importing import read_tabular_file, suggest_mapping, map_sales_data
from src.data.loader import STANDARD_COLUMNS, SalesValidationError
from src.data.import_profiles import list_import_profiles, save_import_profile, missing_profile_columns


LABELS = {"data": "Data *", "produto": "Produto *", "quantidade": "Quantidade *",
          "preco_unitario": "Preço unitário", "faturamento": "Faturamento", "desconto": "Desconto",
          "categoria": "Categoria", "regiao": "Região ou país", "vendedor": "Vendedor"}


@st.cache_data(ttl=600, max_entries=3, show_spinner=False)
def _read_upload(content, filename, separator, encoding, sheet):
    return read_tabular_file(content, filename, separator=separator, encoding=encoding, sheet=sheet)


def render_validation_report(report):
    if not report:
        return
    with st.container(horizontal=True):
        st.metric("Linhas no arquivo", str(report["input_rows"]), border=True)
        st.metric("Linhas válidas", str(report["valid_rows"]), border=True)
        st.metric("Linhas com problemas", str(report["invalid_rows"]), border=True)
    if report.get("missing_columns"):
        st.warning("Colunas obrigatórias ausentes: " + ", ".join(report["missing_columns"]))
    if report.get("duplicate_columns"):
        st.warning("Nomes de colunas duplicados: " + ", ".join(report["duplicate_columns"]))
    for issue in report.get("issues", []):
        positions = ", ".join(map(str, issue["source_rows_sample"]))
        st.warning(f"{issue['column']}: {issue['reason']} {issue['count']} ocorrência(s). Posições de exemplo na tabela: {positions} (contando o cabeçalho como linha 1).")
    for warning in report.get("warnings", []):
        st.caption(f"{warning['count']} registro(s): {warning['message']}")


def render_import_wizard(content, filename, apply_dataset):
    identity = sha256(content).hexdigest()
    prefix = "import_" + identity[:12]
    st.subheader("Prepare seu arquivo", icon=":material/table_view:")
    st.caption("Confira a prévia, associe as colunas e valide antes de carregar. Nenhum dado é enviado à IA nesta preparação.")
    try:
        profiles, invalid_profiles = list_import_profiles()
    except OSError:
        profiles, invalid_profiles = [], 0
        st.warning("Não foi possível ler os perfis salvos. A importação manual continua disponível.")
    if invalid_profiles:
        st.caption(f"{invalid_profiles} perfil(is) inválido(s) foram ignorados.")
    by_id = {item["id"]: item for item in profiles}
    if profiles:
        chosen = st.selectbox("Perfil de importação salvo", [None, *by_id],
            format_func=lambda key: "Configurar manualmente" if key is None else by_id[key]["name"], key=prefix + "_profile")
        if st.button("Aplicar perfil", disabled=chosen is None, key=prefix + "_apply_profile"):
            settings = by_id[chosen]["config"]
            st.session_state[prefix + "_pending_profile"] = settings
            st.session_state[prefix + "_profile_name"] = by_id[chosen]["name"]
            for option in ("separator", "encoding"):
                st.session_state[prefix + "_" + option] = settings["read_options"][option]
            st.session_state.pop(prefix + "_sheet", None)
            st.session_state.pop("import_prepared", None)
            st.session_state.pop("import_error", None)
            st.session_state.pop("import_validation", None)
    pending_profile = st.session_state.get(prefix + "_pending_profile")
    left, right = st.columns(2)
    separator = left.selectbox("Separador do CSV", [",", ";", "\t", "|"],
                               format_func=lambda item: "Tabulação" if item == "\t" else item, key=prefix + "_separator")
    encoding = right.selectbox("Codificação do CSV", ["utf-8-sig", "latin-1"], key=prefix + "_encoding")
    try:
        initial, sheets = _read_upload(content, filename, separator, encoding, None)
        if pending_profile and sheets:
            saved_sheet = pending_profile["read_options"].get("sheet")
            if saved_sheet and saved_sheet not in sheets:
                raise ValueError(f"A aba '{saved_sheet}' do perfil não existe nesta planilha. Escolha a aba manualmente e reaplique as associações.")
            if saved_sheet:
                st.session_state[prefix + "_sheet"] = saved_sheet
        sheet = st.selectbox("Aba da planilha", sheets, key=prefix + "_sheet") if sheets else None
        raw = _read_upload(content, filename, separator, encoding, sheet)[0] if sheet else initial
    except Exception as error:
        st.warning(f"Não foi possível ler este arquivo: {error}")
        st.caption("Confira o formato, o separador e a codificação. Para a UCI, a opção própria preserva as regras dessa fonte.")
        if pending_profile and st.button("Descartar perfil e configurar manualmente", key=prefix + "_discard_profile"):
            st.session_state.pop(prefix + "_pending_profile", None)
            st.rerun()
        return
    st.caption(f"{len(raw):,} linhas · {len(raw.columns)} colunas · prévia das primeiras 12 linhas".replace(",", "."))
    st.dataframe(raw.head(12), hide_index=True, width="stretch")
    signature = f"{identity}:{separator}:{encoding}:{sheet}"
    if st.session_state.get("import_signature") != signature:
        for key in ("import_prepared", "import_error", "import_validation"):
            st.session_state.pop(key, None)
        st.session_state.import_signature = signature
    form_key = prefix + "_" + sha256(signature.encode()).hexdigest()[:8]
    if pending_profile:
        missing = missing_profile_columns(pending_profile, raw.columns)
        if missing:
            st.warning("Este arquivo não tem as colunas do perfil: " + ", ".join(missing) + ". Nenhuma base foi carregada.")
            if st.button("Descartar perfil e configurar manualmente", key=prefix + "_discard_profile"):
                st.session_state.pop(prefix + "_pending_profile", None)
                st.rerun()
            return
        for field, column in pending_profile["mapping"].items():
            st.session_state[form_key + "_" + field] = column
        for option, suffix in (("currency", "currency"), ("date_format", "dates"), ("decimal", "decimal"),
                ("discount_percent", "percent"), ("revenue_mode", "revenue"), ("price_mode", "price"), ("invalid_policy", "policy")):
            st.session_state[form_key + "_" + suffix] = pending_profile[option]
        st.session_state.pop(prefix + "_pending_profile", None)
        st.success("Perfil aplicado. Confira as opções e valide este arquivo antes de carregar.")
        if pending_profile["invalid_policy"] == "drop":
            st.warning("Este perfil propõe excluir linhas inválidas. Confira essa regra e o relatório de validação antes de usar a base.")
    suggestions = suggest_mapping(raw.columns)
    options = [None, *raw.columns.tolist()]
    with st.form(form_key + "_mapping"):
        st.markdown("**Associe as colunas do seu arquivo**")
        mapping = {}
        columns = st.columns(2)
        for index, field in enumerate(STANDARD_COLUMNS):
            mapping[field] = columns[index % 2].selectbox(LABELS[field], options,
                index=None if form_key + "_" + field in st.session_state else options.index(suggestions[field]),
                format_func=lambda value: "Não informado / calculado" if value is None else str(value), key=form_key + "_" + field)
        st.caption("Data, produto e quantidade são obrigatórios. Categoria, região, vendedor e desconto podem ficar indisponíveis. Informe faturamento e preço, ou escolha abaixo como calcular um deles.")
        left, right = st.columns(2)
        currency = left.selectbox("Moeda dos valores", ["BRL", "GBP", "USD", "EUR"], key=form_key + "_currency")
        date_format = right.selectbox("Formato das datas", ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "mixed"],
            format_func=lambda value: {"%Y-%m-%d": "AAAA-MM-DD", "%Y-%m-%d %H:%M:%S": "AAAA-MM-DD HH:MM:SS", "%d/%m/%Y": "DD/MM/AAAA", "%m/%d/%Y": "MM/DD/AAAA", "mixed": "Automático (confira datas ambíguas)"}[value], key=form_key + "_dates")
        decimal = left.selectbox("Separador decimal dos números em texto", [".", ","], key=form_key + "_decimal")
        discount_percent = right.checkbox("Desconto em porcentagem (10 representa 10%)", key=form_key + "_percent")
        revenue_mode = left.selectbox("Como obter o faturamento", ["column", "calculate"],
            format_func=lambda value: "Usar coluna informada" if value == "column" else "Quantidade × preço × (1 − desconto)", key=form_key + "_revenue")
        price_mode = right.selectbox("Como obter o preço", ["column", "calculate"],
            format_func=lambda value: "Usar coluna informada" if value == "column" else "Faturamento ÷ quantidade", key=form_key + "_price")
        st.caption("Ao calcular faturamento sem desconto informado, você assume desconto zero apenas para a fórmula. Isso não transforma o preenchimento em um desconto observado. Não há conversão de moeda.")
        policy = st.selectbox("Se houver linhas com datas ou números inválidos", ["raise", "drop"],
            format_func=lambda value: "Bloquear e mostrar os problemas" if value == "raise" else "Excluir somente as linhas inválidas e registrar a exclusão", key=form_key + "_policy")
        validated = st.form_submit_button("Validar mapeamento", type="primary")
        apply_requested = st.form_submit_button("Usar esta base", key=form_key + "_apply",
            disabled=st.session_state.get("import_prepared", {}).get("signature") != signature)
    if validated or apply_requested:
        st.session_state.pop("import_prepared", None)
        try:
            prepared = map_sales_data(raw, mapping, currency=currency, date_format=date_format, decimal=decimal,
                discount_percent=discount_percent, revenue_mode=revenue_mode, price_mode=price_mode,
                invalid_policy=policy, filename=filename, source_hash=identity)
            audit = prepared.attrs["mapping_audit"]
            profile_config = {key: audit[key] for key in ("mapping", "date_format", "decimal", "discount_percent", "revenue_mode", "price_mode", "invalid_policy", "read_options")}
            profile_config["currency"] = currency
            st.session_state.import_prepared = {"signature": signature, "data": prepared, "profile_config": profile_config}
            st.session_state.import_validation = prepared.attrs["import_validation_report"]
            st.session_state.import_error = None
            if apply_requested and apply_dataset(prepared.copy(), filename) is not False:
                st.session_state.import_applied = identity
        except SalesValidationError as error:
            st.session_state.import_validation = error.report
            st.session_state.import_error = str(error)
        except (ValueError, TypeError) as error:
            st.session_state.import_validation = None
            st.session_state.import_error = str(error)
        st.rerun()
    preview = st.session_state.get("import_prepared")
    if st.session_state.get("import_error"):
        st.warning(st.session_state.import_error)
    if st.session_state.get("import_validation"):
        render_validation_report(st.session_state.get("import_validation"))
    if preview and preview["signature"] == signature:
        st.markdown("**Resultado que será usado na análise**")
        st.dataframe(preview["data"].head(12), hide_index=True, width="stretch")
        st.caption("Confira esta prévia e clique em Usar esta base no formulário acima. As escolhas são validadas novamente ao carregar.")
        with st.expander("Guardar estas configurações como perfil", icon=":material/bookmark:"):
            st.caption("Salva as opções da última validação: colunas, moeda, datas, números e regras de leitura. Os registros de vendas não são incluídos. Se alterar o formulário, valide novamente antes de salvar.")
            with st.form(prefix + "_save_profile_form"):
                name = st.text_input("Nome do perfil", max_chars=80, key=prefix + "_new_profile_name", placeholder="Ex.: Vendas da loja — planilha mensal")
                save_requested = st.form_submit_button("Salvar perfil")
            if save_requested:
                try:
                    save_import_profile(name, preview["profile_config"])
                    st.success("Perfil salvo neste computador. Ele estará disponível na próxima importação.")
                except (OSError, ValueError) as error:
                    st.warning(f"Não foi possível salvar o perfil: {error}")
