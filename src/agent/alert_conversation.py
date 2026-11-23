"""Deterministic answers scoped to one alert, without monthly reinterpretation."""

from datetime import datetime
import re

import pandas as pd

from src.agent.conversational_analytics import normalize_text
from src.agent.period_parser import extract_periods_from_text
from src.data.capabilities import available_dimensions
from src.formatting import format_currency


def alert_records(data, context):
    mask = data["data"].dt.normalize().between(pd.Timestamp(context["start_date"]), pd.Timestamp(context["end_date"]))
    scope = context.get("scope", {})
    if scope.get("dimension") and scope.get("segmento") is not None:
        mask &= data[scope["dimension"]].astype("string").eq(str(scope["segmento"]))
    return data.loc[mask]


def build_alert_context(data, facts, config):
    return {**facts, "created_at": datetime.now().isoformat(), "source": dict(data.attrs.get("dataset_context", {})),
            "scope": {key: config[key] for key in ("dimension", "segmento") if config.get(key) is not None}}


def answer_alert_question(question, data, context):
    text = normalize_text(question)
    source = context.get("source", {})
    dimensions = available_dimensions(data)
    scope = context.get("scope", {})
    label = f"**Alerta de {context['period']}**"
    if scope.get("dimension"):
        label += f" · {scope['dimension']}: {scope.get('segmento')}"
    prefix = label + ". "
    def result(content, tables=None):
        return {"content": prefix + content, "mode": "local", "tables": tables or []}
    # Explicitly reject dates outside the active alert instead of silently ignoring them.
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)
    for date in re.findall(r"\b\d{2}/\d{2}/\d{4}\b", question):
        parsed = pd.to_datetime(date, format="%d/%m/%Y", errors="coerce")
        if pd.notna(parsed):
            dates.append(parsed.date().isoformat())
    anchor = pd.DataFrame({"data": pd.to_datetime([context["start_date"]])})
    months = extract_periods_from_text(question, anchor)
    if any(date < context["start_date"] or date > context["end_date"] for date in dates) or any(month not in (context["start_date"][:7], context["end_date"][:7]) for month in months) or any(word in text for word in ("por mes", "mensal", "outro periodo", "outro dia")):
        return result("Esta conversa está limitada ao intervalo do alerta. Clique em **Sair do alerta** para comparar outros períodos ou abra outro alerta em Analisar.")
    dimension = next((field for field, words in (("produto", ("produto",)), ("categoria", ("categoria",)), ("regiao", ("regiao", "regioes", "pais", "paises")), ("vendedor", ("vendedor",))) if any(word in text for word in words)), None)
    if dimension and dimension not in dimensions:
        return result(f"A fonte não informa {dimension}; o preenchimento dessa coluna não permite analisar esse grupo.")
    if "desconto" in text and source.get("discount_available") is False:
        return result("A fonte não informa o desconto aplicado. Um zero de preenchimento não comprova uma venda sem desconto.")
    if any(word in text for word in ("lucro", "margem", "custo")):
        return result("A base padronizada não informa custos e não permite calcular lucro ou margem. Faturamento é receita de vendas.")
    observed = alert_records(data, context)
    money = lambda number: format_currency(number, source)
    number = lambda value: f"{float(value):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    value = money if context["metric"] == "faturamento" else lambda value: number(value) + " unidades"
    if dimension:
        metric = "faturamento" if "faturamento" in text or "receita" in text else "quantidade" if "quantidade" in text or "unidade" in text else context["metric"]
        grouped = observed.groupby(dimension, dropna=False)[metric].sum()
        smallest = any(word in text for word in ("menor", "menos", "pior"))
        frame = (grouped.nsmallest(8) if smallest else grouped.nlargest(8)).reset_index()
        return result("Estes são os " + ("menores" if smallest else "maiores") + " valores registrados no intervalo selecionado. A distribuição não divide a referência estatística entre os grupos nem confirma uma causa.",
                      [{"title": f"Registros do alerta por {dimension}" + (f" ({source.get('currency', 'BRL')})" if metric == "faturamento" else " (unidades)"),
                        "data": frame.to_dict("records"), "chart": {"x": dimension, "y": metric, "type": "bar"}}])
    if "quantidade" in text or "unidade" in text:
        return result(f"Foram registradas **{number(observed['quantidade'].sum())} unidades** em {len(observed)} linhas, de {context['start_date']} a {context['end_date']}. São registros da base; uma linha não representa necessariamente um pedido.")
    if context["metric"] != "faturamento" and any(word in text for word in ("faturamento", "receita")):
        return result(f"O faturamento registrado no intervalo foi **{money(observed['faturamento'].sum())}**. O alerta selecionado foi calculado para quantidade; sua referência não é uma previsão de faturamento.")
    if any(word in text for word in ("promoc", "feriado", "estoque")):
        if context.get("calendar_context"):
            return result(f"O calendário deste dia informa **{context['calendar_context']}**. A referência já considera o calendário informado; o alerta indica um desvio que permaneceu após esse ajuste. Esse contexto não comprova a causa da diferença restante. Confira os registros e outras evidências do mesmo intervalo.")
        return result("As vendas, isoladamente, não confirmam esse motivo. Confira um calendário comercial, registros de promoções ou estoque para o mesmo intervalo e o mesmo grupo. Mesmo uma coincidência de datas precisa de avaliação antes de concluir causalidade.")
    facts = f"O valor observado foi **{value(context['observed'])}**, diante de uma referência de **{value(context['expected'])}**: diferença de **{value(context['difference'])}**."
    if context.get("percentage") is not None:
        facts += f" A variação em relação à referência foi {number(context['percentage'])}%."
    else:
        facts += " A referência nula ou negativa não permite uma comparação percentual adequada."
    if any(word in text for word in ("limite", "escore", "regra", "referencia", "calcul", "esperad")):
        score, threshold = context.get("score"), context.get("threshold")
        rule = f" Escore: {number(score)}; limite: {number(threshold)}." if score is not None and threshold is not None else ""
        return result(facts + f" A referência usa {context['previous_observations']} observações anteriores. Método de calibração: {context.get('calibration_method') or 'registrado na análise'}." + rule + " O escore não é uma probabilidade de fraude ou de erro.")
    checks = "\n".join("- " + item for item in context.get("next_checks", []))
    return result(facts + "\n\n**O que conferir agora:**\n" + checks + "\n\nEsses números sinalizam uma mudança; ainda não confirmam sua causa. Você pode perguntar pela quantidade, pelos grupos disponíveis ou pela regra do alerta. Para consultar outros períodos, use **Sair do alerta**.")
