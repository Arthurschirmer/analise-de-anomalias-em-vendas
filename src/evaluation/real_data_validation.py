"""Reproducible, offline validation of the local Olist sales dataset.

The evaluation uses an explicit MockLLMProvider, never a configured API. Real
signals have no ground-truth anomaly labels, so this runner reports evidence and
technical checks rather than precision/recall or confirmed causes.
"""

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.agent.investigation_agent import InvestigationAgent
from src.analysis.anomaly_detection import build_anomaly_description, detect_period_anomalies
from src.data.adapters.olist_adapter import OlistAdapter
from src.llm.mock_provider import MockLLMProvider
from src.ui.analytics import monthly_overview
from src.ui.charts import trend_chart


METHODS = ("percentual", "zscore", "iqr", "media_movel", "combinado")
METHOD_LABELS = {
    "percentual": "Variação percentual", "zscore": "Z-score", "iqr": "IQR",
    "media_movel": "Média móvel", "combinado": "Combinado",
}
SEMANTICS = [
    "Somente pedidos com status delivered são incluídos; o recorte é retrospectivo.",
    "O mês se refere à data de compra, não à data de entrega ou de reconhecimento contábil.",
    "Faturamento é a soma de price dos produtos; exclui frete, custos e lucro.",
    "Desconto zero é preenchimento do adaptador porque não há coluna de desconto; não comprova ausência de descontos.",
    "O campo regiao representa a UF do cliente; produto e vendedor são identificadores anonimizados.",
    "As linhas padronizadas agregam data/produto/categoria/UF/vendedor; não são contagens de pedidos.",
    "Preço médio é uma média por registro agregado; composição de produtos e agrupamento podem influenciá-la.",
    "Dias sem vendas não comprovam ausência de dados. Cobertura observada não certifica completude.",
    "O período selecionado é informado nos argumentos da execução; não é ajustado automaticamente para maximizar detecções.",
    "Sinalizações são hipóteses para investigação; não demonstram erro, fraude, significância estatística ou causalidade.",
    "Sem rótulos reais de anomalia ou causa, não se calculam acurácia, precisão ou recall.",
    "A execução usa MockLLMProvider local e não avalia a qualidade nem a conectividade de um modelo remoto.",
]
GROUP_KEYS = ["data", "produto", "categoria", "regiao", "vendedor"]


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if value is None:
        return None
    if isinstance(value, (datetime, pd.Timestamp, pd.Period)):
        return str(value)
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _write_json(path, value):
    path.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _period(value):
    if value is None:
        return None
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ValueError("Os períodos devem usar o formato AAAA-MM.")
    return str(pd.Period(value, freq="M"))


def _require_columns(frame, names, filename):
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{filename}: faltam colunas obrigatórias: {', '.join(sorted(missing))}")


def _numeric_quality(series):
    numbers = pd.to_numeric(series, errors="coerce")
    finite = numbers.map(lambda value: pd.notna(value) and math.isfinite(float(value)))
    return {
        "ausentes_ou_invalidos": int((~finite).sum()),
        "negativos": int(numbers.lt(0).sum()),
        "zeros": int(numbers.eq(0).sum()),
    }


def audit_olist_source(input_dir):
    """Audit keys, joins, coverage and conservation without exposing customers."""
    source = Path(input_dir)
    paths = {name: source / filename for name, filename in OlistAdapter.REQUIRED_FILES.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Arquivos obrigatórios da Olist não encontrados: {', '.join(missing)}")
    columns = {
        "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
        "items": ["order_id", "order_item_id", "product_id", "seller_id", "price"],
        "products": ["product_id", "product_category_name"],
        "customers": ["customer_id", "customer_state"],
    }
    raw = {}
    for name, path in paths.items():
        frame = pd.read_csv(path)
        _require_columns(frame, columns[name], path.name)
        extra = ["freight_value"] if name == "items" and "freight_value" in frame else []
        raw[name] = frame[columns[name] + extra].copy()

    orders, items = raw["orders"], raw["items"]
    products, customers = raw["products"], raw["customers"]
    keys = {
        "orders": ["order_id"], "items": ["order_id", "order_item_id"],
        "products": ["product_id"], "customers": ["customer_id"],
    }
    duplicates = {name: int(raw[name].duplicated(key).sum()) for name, key in keys.items()}
    null_keys = {name: int(raw[name][key].isna().any(axis=1).sum()) for name, key in keys.items()}
    required_nulls = {
        name: frame[[column for column in columns[name] if column != "product_category_name"]].isna().sum().astype(int).to_dict()
        for name, frame in raw.items()
    }
    orphans = {
        "itens_sem_pedido": int((~items.order_id.isin(orders.order_id)).sum()),
        "itens_sem_produto": int((~items.product_id.isin(products.product_id)).sum()),
        "pedidos_sem_cliente": int((~orders.customer_id.isin(customers.customer_id)).sum()),
    }
    numeric = {"price": _numeric_quality(items.price)}
    if "freight_value" in items:
        numeric["freight_value"] = _numeric_quality(items.freight_value)
    dates = pd.to_datetime(orders.order_purchase_timestamp, errors="coerce")
    orders = orders.assign(timestamp=dates)
    delivered = orders.loc[orders.order_status.eq("delivered")].copy()
    structural_errors = []
    if any(duplicates.values()) or any(null_keys.values()):
        structural_errors.append("Há chaves duplicadas ou ausentes que impedem joins confiáveis.")
    if any(orphans.values()):
        structural_errors.append("Há relacionamentos órfãos entre os arquivos.")
    if dates.isna().any():
        structural_errors.append("Há datas de compra ausentes ou inválidas.")
    if any(value for fields in required_nulls.values() for value in fields.values()):
        structural_errors.append("Há valores obrigatórios ausentes.")
    if any(q["ausentes_ou_invalidos"] or q["negativos"] for q in numeric.values()):
        structural_errors.append("Há valores monetários inválidos ou negativos.")
    if delivered.empty:
        structural_errors.append("Não há pedidos entregues para analisar.")
    if structural_errors:
        raise ValueError("Falha estrutural na base Olist: " + " ".join(structural_errors))

    joined = (
        items.merge(delivered[["order_id", "customer_id", "timestamp"]], on="order_id", how="inner", validate="many_to_one")
        .merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(customers, on="customer_id", how="left", validate="many_to_one")
    )
    missing_delivered_items = int((~delivered.order_id.isin(items.order_id)).sum())
    if missing_delivered_items:
        raise ValueError("Há pedidos entregues sem itens correspondentes.")

    translation_path = source / OlistAdapter.TRANSLATION_FILE
    untranslated = []
    translation_duplicates = 0
    if translation_path.is_file():
        translation = pd.read_csv(translation_path)
        _require_columns(translation, ["product_category_name", "product_category_name_english"], translation_path.name)
        translation_duplicates = int(translation.duplicated("product_category_name").sum())
        if translation_duplicates:
            raise ValueError("A tradução de categorias contém chaves duplicadas que ampliariam os joins.")
        untranslated = sorted(products.loc[
            products.product_category_name.notna() & ~products.product_category_name.isin(translation.product_category_name),
            "product_category_name",
        ].unique().tolist())

    standardized = OlistAdapter(source).load()
    raw_revenue = float(pd.to_numeric(joined.price).sum())
    standard_revenue = float(standardized.faturamento.sum())
    checks = {
        "chaves_sem_duplicatas": not any(duplicates.values()) and translation_duplicates == 0,
        "relacionamentos_sem_orfaos": not any(orphans.values()),
        "quantidade_conservada": int(standardized.quantidade.sum()) == len(joined),
        "faturamento_conservado": math.isclose(standard_revenue, raw_revenue, abs_tol=0.000001, rel_tol=0),
        "chaves_padronizadas_unicas": not standardized.duplicated(GROUP_KEYS).any(),
        "sem_nulos_padronizados": not standardized.isna().any().any(),
        "faturamento_igual_preco_vezes_quantidade": bool(
            ((standardized.preco_unitario * standardized.quantidade - standardized.faturamento).abs() <= 0.000001).all()
        ),
    }
    if not all(checks.values()):
        raise ValueError("A reconciliação da base padronizada falhou: " + ", ".join(name for name, passed in checks.items() if not passed))

    joined["periodo"] = joined.timestamp.dt.to_period("M").astype(str)
    monthly = joined.groupby("periodo", as_index=False).agg(
        pedidos_entregues=("order_id", "nunique"),
        itens=("order_item_id", "size"),
        faturamento=("price", "sum"),
        primeira_compra=("timestamp", "min"),
        ultima_compra=("timestamp", "max"),
        dias_com_vendas=("timestamp", lambda values: values.dt.normalize().nunique()),
    )
    monthly["dias_no_mes"] = monthly.periodo.map(lambda value: pd.Period(value, freq="M").days_in_month)
    monthly["cobertura_dias_com_vendas_percentual"] = monthly.dias_com_vendas / monthly.dias_no_mes * 100
    monthly["observacao_nao_cobre_extremos_do_mes"] = (
        monthly.primeira_compra.dt.day.ne(1) | monthly.ultima_compra.dt.day.ne(monthly.dias_no_mes)
    )
    all_months = pd.period_range(monthly.periodo.min(), monthly.periodo.max(), freq="M").astype(str)
    absent = sorted(set(all_months) - set(monthly.periodo))
    monthly["mes_de_borda"] = monthly.periodo.isin([monthly.periodo.min(), monthly.periodo.max()])
    monthly["primeira_compra"] = monthly.primeira_compra.dt.strftime("%Y-%m-%d %H:%M:%S")
    monthly["ultima_compra"] = monthly.ultima_compra.dt.strftime("%Y-%m-%d %H:%M:%S")
    no_category = standardized.loc[standardized.categoria.eq("sem_categoria")]
    warnings = []
    if not no_category.empty:
        warnings.append("Há produtos sem categoria; suas vendas foram preservadas em sem_categoria.")
    if absent:
        warnings.append("Há meses sem registros entre os meses observados; não devem ser tratados automaticamente como vendas zero.")
    if monthly.observacao_nao_cobre_extremos_do_mes.any():
        warnings.append("Alguns meses não têm vendas observadas nos extremos do mês. Esse indicador exige revisão de cobertura, não comprova incompletude.")
    if untranslated:
        warnings.append("Algumas categorias não têm tradução e mantêm o rótulo original.")
    used_names = set(OlistAdapter.REQUIRED_FILES.values()) | {OlistAdapter.TRANSLATION_FILE}
    provenance = [{
        "arquivo": path.name, "bytes": path.stat().st_size, "sha256": _hash_file(path),
        "usado_pelo_adaptador": path.name in used_names,
    } for path in sorted(source.glob("*.csv"))]
    audit = {
        "fonte": "Olist local; versão identificada pelos hashes dos arquivos",
        "diretorio": str(source.resolve()),
        "arquivos": provenance,
        "contagens_brutas": {name: len(frame) for name, frame in raw.items()},
        "status_pedidos": orders.order_status.value_counts().astype(int).to_dict(),
        "duplicatas_chaves": {**duplicates, "translation": translation_duplicates},
        "chaves_ausentes": null_keys, "nulos_obrigatorios": required_nulls,
        "orfaos": orphans, "qualidade_numerica": numeric,
        "datas_brutas": {"minima": str(dates.min()), "maxima": str(dates.max()), "invalidas": int(dates.isna().sum())},
        "pedidos_entregues": len(delivered),
        "itens_entregues": len(joined),
        "pedidos_entregues_sem_itens": missing_delivered_items,
        "itens_excluidos_por_status": len(items) - len(joined),
        "linhas_padronizadas": len(standardized),
        "quantidade_padronizada": int(standardized.quantidade.sum()),
        "faturamento_padronizado": standard_revenue,
        "faturamento_itens_entregues": raw_revenue,
        "diferenca_faturamento": standard_revenue - raw_revenue,
        "frete_excluido": float(joined.freight_value.sum()) if "freight_value" in joined else None,
        "categorias_ausentes": {
            "produtos_brutos": int(products.product_category_name.isna().sum()),
            "linhas_padronizadas": len(no_category),
            "itens_entregues": int(no_category.quantidade.sum()),
            "faturamento": float(no_category.faturamento.sum()),
        },
        "categorias_sem_traducao": untranslated,
        "dimensoes": {column: int(standardized[column].nunique()) for column in ["produto", "categoria", "regiao", "vendedor"]},
        "periodos_sem_registros": absent,
        "verificacoes": checks, "avisos": warnings,
    }
    return _clean(audit), standardized, monthly


def _consecutive(item):
    return pd.Period(item["periodo_atual"], freq="M") == pd.Period(item["periodo_anterior"], freq="M") + 1


def _detect(data, method, threshold, minimum):
    return [
        row for row in detect_period_anomalies(
            data, metric="faturamento", method=method,
            threshold_pct=threshold, min_period_value=minimum,
        ) if _consecutive(row)
    ]


def _money(value):
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(value):
    return "indisponível" if value is None else f"{float(value):.2f}%".replace(".", ",")


def _scope_summary(data):
    observed = sorted(data.data.dt.to_period("M").astype(str).unique())
    return {
        "inicio": observed[0], "fim": observed[-1], "meses_observados": len(observed),
        "linhas": len(data), "quantidade": int(data.quantidade.sum()),
        "faturamento": float(data.faturamento.sum()),
    }


def run_real_data_validation(
    input_dir="data/olist_raw", output_dir="outputs/real_data/olist",
    start_period=None, end_period=None, threshold_pct=30.0,
    min_period_value=1000.0, max_investigations=5, selection_note=None,
):
    """Run aggregate validation, five detectors and deterministic investigations."""
    start_period, end_period = _period(start_period), _period(end_period)
    if start_period and end_period and start_period > end_period:
        raise ValueError("O início do período deve ser anterior ou igual ao fim.")
    if not math.isfinite(threshold_pct) or not math.isfinite(min_period_value) or threshold_pct < 0 or min_period_value < 0 or max_investigations < 0:
        raise ValueError("Limiar e valor mínimo devem ser finitos e não negativos; o limite de investigações deve ser não negativo.")
    started = time.perf_counter()
    audit, full, monthly = audit_olist_source(input_dir)
    month_values = full.data.dt.to_period("M").astype(str)
    mask = pd.Series(True, index=full.index)
    if start_period:
        mask &= month_values >= start_period
    if end_period:
        mask &= month_values <= end_period
    selected = full.loc[mask].copy()
    if selected.empty:
        raise ValueError("O período escolhido não possui vendas entregues na base.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output / "report.md", "summary": output / "summary.json",
        "monthly_summary": output / "monthly_summary.csv",
        "detector_comparison": output / "detector_comparison.csv",
        "sensitivity": output / "sensitivity.csv",
        "selected_standardized": output / "selected_standardized.csv",
        "monthly_full_chart": output / "monthly_full.html",
        "monthly_selected_chart": output / "monthly_selected.html",
    }
    records, detector_summary = [], []
    for scope, data in [("integral", full), ("selecionado", selected)]:
        for method in METHODS:
            rows = _detect(data, method, threshold_pct, min_period_value)
            for row in rows:
                records.append({"escopo": scope, **row})
            detector_summary.append({
                "escopo": scope, "metodo": method,
                "comparacoes_consecutivas": len(rows),
                "comparacoes_com_base_minima": sum(bool(row["base_suficiente"]) for row in rows),
                "sinais": sum(bool(row["anomalia"]) for row in rows),
            })

    sensitivity = []
    for threshold in sorted({20.0, 30.0, 40.0, float(threshold_pct)}):
        rows = _detect(selected, "percentual", threshold, min_period_value)
        flagged = [row for row in rows if row["anomalia"]]
        sensitivity.append({
            "escopo": "selecionado", "metodo": "percentual", "limite_percentual": threshold,
            "valor_minimo_periodo": min_period_value, "comparacoes_consecutivas": len(rows),
            "sinais": len(flagged),
            "periodos_sinalizados": "; ".join(row["periodo_atual"] for row in flagged),
        })

    primary = _detect(selected, "percentual", threshold_pct, min_period_value)
    flagged = sorted(
        [row for row in primary if row["anomalia"]],
        key=lambda row: abs(row["variacao_percentual"] or 0), reverse=True,
    )
    investigations = []
    agent = InvestigationAgent(llm_provider=MockLLMProvider())
    for anomaly in flagged[:max_investigations]:
        case_started = time.perf_counter()
        investigation = agent.investigate(
            build_anomaly_description(anomaly), selected, extra_evidence=anomaly,
        )
        name = f"{anomaly['periodo_atual']}_vs_{anomaly['periodo_anterior']}"
        case_path = output / "investigations" / f"{name}.json"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(case_path, investigation.to_dict())
        investigations.append({
            "id": name, "arquivo": str(case_path.resolve()),
            "periodo_anterior": anomaly["periodo_anterior"], "periodo_atual": anomaly["periodo_atual"],
            "resumo_comparativo": investigation.structured_report.get("resumo_comparativo", {}),
            "causas_priorizadas": investigation.structured_report.get("causas_provaveis", []),
            "tempo_segundos": round(time.perf_counter() - case_started, 4),
        })

    monthly.to_csv(paths["monthly_summary"], index=False, encoding="utf-8-sig")
    detector_frame = pd.DataFrame(_clean(records))
    if detector_frame.empty:
        detector_frame = pd.DataFrame(columns=["escopo", "metodo", "periodo_anterior", "periodo_atual", "anomalia"])
    detector_frame.to_csv(paths["detector_comparison"], index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(paths["sensitivity"], index=False, encoding="utf-8-sig")
    selected.to_csv(paths["selected_standardized"], index=False, encoding="utf-8-sig")
    for scope, frame, chart_key in [
        ("Histórico integral", full, "monthly_full_chart"),
        ("Recorte selecionado", selected, "monthly_selected_chart"),
    ]:
        chart_monthly = monthly_overview(frame, threshold_pct=threshold_pct)
        eligible_flags = {
            row["periodo_atual"] for row in _detect(frame, "percentual", threshold_pct, min_period_value)
            if row["anomalia"]
        }
        chart_monthly["anomalia"] = chart_monthly.periodo.isin(eligible_flags)
        chart = trend_chart(chart_monthly).properties(
            title=f"{scope} · variação ≥ {threshold_pct:g}% · piso {_money(min_period_value)}"
        )
        # Altair embeds aggregate values and CDN URLs; generation makes no request.
        chart.save(str(paths[chart_key]), inline=False)
    summary = _clean({
        "versao_validacao": 1,
        "executado_em_utc": datetime.now(timezone.utc).isoformat(),
        "modo": "local_mock_sem_api",
        "fonte_publica": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "parametros": {
            "inicio_solicitado": start_period, "fim_solicitado": end_period,
            "limite_percentual": threshold_pct, "valor_minimo_periodo": min_period_value,
            "maximo_investigacoes": max_investigations,
            "justificativa_recorte": selection_note,
            "meses_media_movel": 3, "limite_zscore": 2.0, "multiplicador_iqr": 1.5,
        },
        "auditoria": audit,
        "contexto_da_base": full.attrs.get("dataset_context", {}),
        "escopos": {"integral": _scope_summary(full), "selecionado": _scope_summary(selected)},
        "detetores": detector_summary,
        "sensibilidade": sensitivity,
        "sinais_percentuais_selecionados": flagged,
        "investigacoes": investigations,
        "semantica_e_limitacoes": SEMANTICS,
        "ground_truth": {"disponivel": False, "metricas_acuracia_calculadas": False},
        "tempo_total_segundos": round(time.perf_counter() - started, 4),
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    })
    paths["report"].write_text(_build_report(summary, monthly), encoding="utf-8")
    _write_json(paths["summary"], summary)
    return summary


def _build_report(summary, monthly):
    audit = summary["auditoria"]
    selected = summary["escopos"]["selecionado"]
    params = summary["parametros"]
    lines = [
        "# Validação exploratória da base real Olist", "",
        "Execução local e reproduzível, com MockLLMProvider explícito e sem chamadas de API.", "",
        f"Fonte: [Brazilian E-Commerce Public Dataset by Olist]({summary['fonte_publica']}).", "",
        "## Recorte e parâmetros", "",
        f"- Recorte solicitado: {params['inicio_solicitado'] or 'início disponível'} a {params['fim_solicitado'] or 'fim disponível'}.",
        f"- Recorte observado: {selected['inicio']} a {selected['fim']}; {selected['meses_observados']} meses.",
        f"- Limiar percentual: {_pct(params['limite_percentual'])}; piso nos dois meses: {_money(params['valor_minimo_periodo'])}.",
        f"- Justificativa do recorte: {params.get('justificativa_recorte') or 'Período informado nos argumentos; sem justificativa adicional registrada.'}",
        "- A seleção é explícita e não é ajustada automaticamente para maximizar resultados.",
        f"- Base selecionada: {selected['linhas']:,} linhas agregadas, {selected['quantidade']:,} itens e {_money(selected['faturamento'])}.", "",
        "## Qualidade e reconciliação", "",
        f"- Pedidos brutos: {audit['contagens_brutas']['orders']:,}; itens brutos: {audit['contagens_brutas']['items']:,}.",
        f"- Pedidos entregues: {audit['pedidos_entregues']:,}; itens entregues: {audit['itens_entregues']:,}.",
        f"- Linhas padronizadas: {audit['linhas_padronizadas']:,}; faturamento: {_money(audit['faturamento_padronizado'])}.",
        f"- Diferença contra a soma bruta dos itens entregues: {_money(audit['diferenca_faturamento'])}.",
        f"- Frete excluído: {_money(audit['frete_excluido']) if audit['frete_excluido'] is not None else 'não disponível no arquivo'}.",
        f"- Itens sem categoria preservados: {audit['categorias_ausentes']['itens_entregues']:,}; {_money(audit['categorias_ausentes']['faturamento'])}.",
        f"- Meses sem registros entre extremos: {', '.join(audit['periodos_sem_registros']) or 'nenhum'}.", "",
    ]
    for name, passed in audit["verificacoes"].items():
        lines.append(f"- {name}: {'OK' if passed else 'FALHOU'}.")
    lines.extend(["", "### Pontos para revisão", ""])
    lines.extend(f"- {warning}" for warning in audit["avisos"])
    lines.extend([
        "", "## Gráficos interativos", "",
        "- [Histórico integral](monthly_full.html)",
        "- [Recorte selecionado](monthly_selected.html)",
        "",
        "Os pontos sinalizados respeitam o limiar e o piso declarados. Os HTMLs contêm apenas agregações mensais; a geração é offline e a exibição carrega as bibliotecas Vega por CDN.",
        "", "## Cobertura mensal observada", "",
        "A tabela mostra somente meses observados. Dias sem vendas não provam arquivo incompleto; extremos esparsos exigem revisão antes de interpretar sinais.", "",
        "| Mês | Pedidos entregues | Itens | Faturamento | Dias com vendas / calendário | Primeira / última data |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in monthly.to_dict("records"):
        lines.append(
            f"| {row['periodo']} | {row['pedidos_entregues']} | {row['itens']} | {_money(row['faturamento'])} | "
            f"{row['dias_com_vendas']} / {row['dias_no_mes']} | {row['primeira_compra'][:10]} / {row['ultima_compra'][:10]} |"
        )
    lines.extend([
        "", "## Comparação dos detectores", "",
        "Todos usam os parâmetros declarados acima. Comparações não consecutivas são excluídas; períodos com piso insuficiente não podem gerar sinal. Os métodos estatísticos exigem histórico próprio e podem responder à tendência da série.", "",
        "| Escopo | Método | Comparações consecutivas | Com piso mínimo | Sinais |",
        "|---|---|---:|---:|---:|",
    ])
    for row in summary["detetores"]:
        lines.append(f"| {row['escopo']} | {METHOD_LABELS[row['metodo']]} | {row['comparacoes_consecutivas']} | {row['comparacoes_com_base_minima']} | {row['sinais']} |")
    lines.extend([
        "", "## Sensibilidade do limiar no recorte selecionado", "",
        "| Limiar | Sinais | Meses sinalizados |", "|---:|---:|---|",
    ])
    for row in summary["sensibilidade"]:
        lines.append(f"| {_pct(row['limite_percentual'])} | {row['sinais']} | {row['periodos_sinalizados'] or 'nenhum'} |")
    lines.extend(["", "## Casos investigados com evidências locais", ""])
    if not summary["investigacoes"]:
        lines.append("Nenhuma comparação ultrapassou o limiar com os critérios atuais.")
    for case in summary["investigacoes"]:
        comparison = case["resumo_comparativo"]
        lines.extend([
            f"### {case['periodo_anterior']} → {case['periodo_atual']}", "",
            f"Faturamento de {_money(comparison['faturamento_anterior'])} para {_money(comparison['faturamento_atual'])}: diferença de {_money(comparison['variacao_faturamento'])} ({_pct(comparison['variacao_percentual_faturamento'])}).",
            f"Quantidade: {comparison['quantidade_anterior']:,} → {comparison['quantidade_atual']:,}.", "",
            "| Dimensão | Segmento | Diferença de faturamento | Diferença de unidades |",
            "|---|---|---:|---:|",
        ])
        for cause in case["causas_priorizadas"][:3]:
            metrics = cause["metricas"]
            lines.append(f"| {cause['dimensao']} | {cause['valor']} | {_money(metrics['variacao_faturamento'])} | {metrics['variacao_quantidade']:.0f} |")
        lines.extend([
            "", "O ranking prioriza associações por impacto; segmentos de dimensões distintas compartilham vendas e suas contribuições não devem ser somadas.",
            f"Registro estruturado: {Path(case['arquivo']).relative_to(Path(summary['outputs']['report']).parent)}.", "",
        ])
    lines.extend(["## Semântica e limites", ""])
    lines.extend(f"- {item}" for item in summary["semantica_e_limitacoes"])
    lines.extend([
        "", "## Arquivos e proveniência", "",
        "O CSV exportado contém o formato padronizado selecionado, sem customer_id, order_id, cidade, CEP ou geolocalização. Produto e vendedor continuam como identificadores anonimizados. O CSV isolado não guarda metadados: as definições da fonte estão neste relatório e em summary.json. No site, prefira Olist local para preservar essa semântica.", "",
        "| Arquivo de entrada | Bytes | SHA-256 | Usado no adaptador |",
        "|---|---:|---|---|",
    ])
    for row in audit["arquivos"]:
        lines.append(f"| {row['arquivo']} | {row['bytes']} | {row['sha256']} | {'sim' if row['usado_pelo_adaptador'] else 'não'} |")
    lines.extend(["", "## Reprodução", "", "Execute validate_real_data.py com os mesmos arquivos, período e parâmetros registrados em summary.json.", ""])
    return "\n".join(lines)

