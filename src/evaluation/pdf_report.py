"""Portable investigation exports built from calculated results, without APIs."""

from datetime import datetime
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

import reportlab
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.formatting import format_currency


INK = colors.HexColor("#17243b")
MUTED = colors.HexColor("#526177")
TEAL = colors.HexColor("#178779")
VIOLET = colors.HexColor("#6954c8")
LIGHT = colors.HexColor("#f0f4f8")
WIDTH = A4[0] - 88


@lru_cache(maxsize=1)
def _fonts():
    folder = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("SalesVera", str(folder / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("SalesVeraBold", str(folder / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("SalesVera", normal="SalesVera", bold="SalesVeraBold", italic="SalesVera", boldItalic="SalesVeraBold")


def _number(value):
    try:
        return f"{float(value):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") if math.isfinite(float(value)) else "Indisponível"
    except (TypeError, ValueError):
        return "Indisponível"


def _plain(value):
    return re.sub(r"[*`#]", "", str(value or "")).replace("→", "para")


def _date_label(value):
    try:
        return datetime.fromisoformat(value).strftime("%d/%m/%Y às %H:%M:%S")
    except (ValueError, TypeError):
        return str(value or "não informado")


class _Report:
    def __init__(self, title, source, created_at, mode):
        _fonts()
        self.parts = []
        self.title = title
        self.styles = {
            "body": ParagraphStyle("body", fontName="SalesVera", fontSize=9, leading=14, textColor=INK, spaceAfter=7, splitLongWords=True),
            "title": ParagraphStyle("title", fontName="SalesVeraBold", fontSize=22, leading=27, textColor=INK, spaceAfter=14),
            "heading": ParagraphStyle("heading", fontName="SalesVeraBold", fontSize=12, leading=16, textColor=TEAL, spaceBefore=12, spaceAfter=8, keepWithNext=True),
            "small": ParagraphStyle("small", fontName="SalesVera", fontSize=8, leading=12, textColor=MUTED, spaceAfter=6),
        }
        self.add("ANÁLISE DE ANOMALIAS EM VENDAS", "small")
        self.add(title, "title")
        self.add("Fonte: " + (source or "Base de vendas"))
        self.add("Registro da análise: " + _date_label(created_at) + " · Modo: " + ("explicação da IA e cálculos locais" if mode == "ai" else "análise local"), "small")

    def paragraph(self, text, style="body"):
        return Paragraph(escape(_plain(text)).replace("\n", "<br/>"), self.styles[style])

    def add(self, text, style="body"):
        self.parts.append(self.paragraph(text, style))

    def table(self, rows, widths=None):
        cells = [[self.paragraph(str(cell), "small") for cell in row] for row in rows]
        table = Table(cells, colWidths=widths or [WIDTH * .58, WIDTH * .42], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#dce4ed")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                                   ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        self.parts.extend([table, Spacer(1, 6)])

    def chart(self, labels, values, formatter):
        if not values or any(value is None or not math.isfinite(float(value)) for value in values):
            return
        values = [float(value) for value in values]
        chart = Drawing(WIDTH, 42 * len(values) + 16)
        scale = max(max(abs(value) for value in values), 1)
        negative = min(values) < 0
        origin = 220 if negative else 145
        span = 92 if negative else 178
        chart.add(Line(origin, 6, origin, chart.height - 6, strokeColor=colors.HexColor("#cbd5e1")))
        for index, (label, value) in enumerate(zip(labels, values)):
            y = chart.height - 32 - index * 42
            short = str(label)
            while pdfmetrics.stringWidth(short, "SalesVera", 8) > 132:
                short = short[:-2] + "…"
            chart.add(String(0, y + 6, short, fontName="SalesVera", fontSize=8, fillColor=INK))
            length = abs(value) / scale * span
            chart.add(Rect(origin - length if value < 0 else origin, y, length, 20, fillColor=TEAL if index % 2 else VIOLET, strokeColor=None))
            chart.add(String(WIDTH - 3, y + 6, formatter(value), fontName="SalesVera", fontSize=8, fillColor=INK, textAnchor="end"))
        self.parts.append(KeepTogether([chart, Spacer(1, 6)]))

    def finish(self):
        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4, leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=48,
                                title=self.title, author="Agente de investigação de anomalias")
        def footer(canvas, document):
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#dce4ed"))
            canvas.line(44, 35, A4[0] - 44, 35)
            canvas.setFont("SalesVera", 7)
            canvas.setFillColor(MUTED)
            canvas.drawString(44, 23, "Evidências de vendas · associações não comprovam causalidade")
            canvas.drawRightString(A4[0] - 44, 23, f"Página {document.page}")
            canvas.restoreState()
        doc.build(self.parts, onFirstPage=footer, onLaterPages=footer)
        return output.getvalue()


def build_investigation_pdf(investigation, *, mode="local", conclusion=None, source_label=None):
    report = investigation.structured_report
    context = report.get("contexto_da_base", {})
    pdf = _Report("Relatório da investigação", context.get("source_label") or source_label, investigation.created_at, mode)
    periods, summary = report.get("periodos", {}), report.get("resumo_comparativo", {})
    before, after = periods.get("periodo_anterior", "não informado"), periods.get("periodo_atual", "não informado")
    pdf.add(f"Comparação: {before} → {after}", "heading")
    pdf.add("Moeda: " + context.get("currency", "BRL") + ". Os valores são apresentados na moeda da fonte, sem conversão.", "small")
    window = context.get("comparison_window", {})
    window_note = context.get("comparison_window_definition")
    if window.get("mode") == "aligned_days":
        pdf.add(f"Recorte de dias: {window['first_day']} a {window['last_day']} nos dois meses.")
    if window_note:
        pdf.add(window_note)
    if context.get("coverage"):
        pdf.add("A presença de registros não comprova cobertura completa do calendário. Meses nas bordas da base podem conter apenas parte das datas.", "small")
    money = lambda value: format_currency(value, context)
    pct = summary.get("variacao_percentual_faturamento")
    pdf.table([["Indicador", "Resultado"], [f"Faturamento em {before}", money(summary.get("faturamento_anterior"))],
               [f"Faturamento em {after}", money(summary.get("faturamento_atual"))],
               ["Diferença de faturamento", money(summary.get("variacao_faturamento"))],
               ["Variação percentual", _number(pct) + "%" if pct is not None else "Indisponível"],
               [f"Quantidade em {before}", _number(summary.get("quantidade_anterior")) + " unidades"],
               [f"Quantidade em {after}", _number(summary.get("quantidade_atual")) + " unidades"]])
    pdf.chart([before, after], [summary.get("faturamento_anterior"), summary.get("faturamento_atual")], money)
    pdf.add("Explicação da investigação", "heading")
    pdf.add(conclusion or investigation.final_conclusion or "Consulte os indicadores calculados.")
    causes = report.get("causas_provaveis", [])[:5]
    if causes:
        pdf.add("Grupos com maior mudança", "heading")
        pdf.add("As dimensões podem compartilhar as mesmas vendas. Não some as linhas como contribuições independentes. O ranking não mede probabilidade causal.", "small")
        pdf.table([["Grupo e dimensão", "Diferença de faturamento"], *[[f"{cause.get('valor')} ({cause.get('dimensao')})", money(cause.get("metricas", {}).get("variacao_faturamento"))] for cause in causes]])
    decomposition = report.get("decomposicao_faturamento", {})
    if decomposition.get("available"):
        pdf.add("Decomposição do faturamento", "heading")
        pdf.add(decomposition.get("convention", ""), "small")
        pdf.table([["Componente", "Valor"], *[[item["label"], money(item["value"])] for item in decomposition.get("components", [])]])
    pdf.add("Definições e limitações", "heading")
    for key in ("row_definition", "revenue_definition", "quantity_definition", "price_definition"):
        if context.get(key):
            pdf.add(context[key])
    if context.get("discount_available") is False:
        pdf.add("Descontos não informados pela fonte. Zeros de preenchimento não representam descontos observados.")
    unavailable = [label for key, label in (("category_available", "categoria"), ("seller_available", "vendedor")) if context.get(key) is False]
    if unavailable:
        pdf.add("Grupos indisponíveis: " + ", ".join(unavailable) + ".")
    limits = dict.fromkeys([*context.get("limitations", []), *report.get("limitacoes", []),
                           "As evidências mostram associações; não confirmam causalidade. Estoque, campanhas e feriados exigem informações adicionais."])
    for limit in limits:
        pdf.add("• " + limit, "small")
    return pdf.finish()


def build_alert_pdf(context, groups=None):
    source = context.get("source", {})
    pdf = _Report("Ficha de investigação do alerta", source.get("source_label"), context.get("created_at"), "local")
    pdf.add("Intervalo: " + context["start_date"] + " a " + context["end_date"], "heading")
    scope = context.get("scope", {})
    if scope.get("dimension"):
        pdf.add(f"Grupo analisado: {scope['dimension']} = {scope.get('segmento')}")
    metric = context["metric"]
    pdf.add("Indicador: " + metric + (" · Moeda: " + source.get("currency", "BRL") + " (sem conversão)" if metric == "faturamento" else " (unidades)"))
    value = (lambda number: format_currency(number, source)) if metric == "faturamento" else (lambda number: _number(number) + " unidades")
    pdf.table([["Indicador", "Resultado"], ["Valor observado", value(context["observed"])],
               ["Referência esperada", value(context["expected"])], ["Diferença da referência", value(context["difference"])],
               ["Variação percentual", _number(context["percentage"]) + "%" if context.get("percentage") is not None else "Indisponível para referência nula ou negativa"],
               ["Observações anteriores", str(context.get("previous_observations", "não informado"))],
               ["Escore / limite", _number(context.get("score")) + " / " + _number(context.get("threshold"))]])
    pdf.chart(["Referência", "Observado"], [context["expected"], context["observed"]], value)
    pdf.add("Como interpretar", "heading")
    pdf.add(f"Método de calibração: {context.get('calibration_method') or 'não informado'}. O escore e o limite são critérios de decisão, não probabilidades de fraude ou de erro.")
    if context.get("calendar_context"):
        pdf.add("Calendário informado: " + context["calendar_context"] + ". A referência já considera esse contexto; ele não comprova a causa do desvio restante.")
    if groups is not None and not groups.empty:
        pdf.add("Registros observados no intervalo", "heading")
        pdf.table([list(groups.columns), *groups.head(8).astype(str).values.tolist()])
        pdf.add("Esta distribuição não divide a referência estatística entre grupos nem confirma quem causou a mudança.", "small")
    pdf.add("O que conferir", "heading")
    for check in context.get("next_checks", []):
        pdf.add("• " + check)
    pdf.add("Limitações", "heading")
    for limit in [*source.get("limitations", []), "Uma anomalia estatística pede verificação; não comprova erro, fraude ou uma causa comercial."]:
        pdf.add("• " + limit, "small")
    return pdf.finish()
