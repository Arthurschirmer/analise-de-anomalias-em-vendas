import json
import re
import unicodedata


def evaluate_investigation(investigation, scenario, top_k=3):
    expected_cause = scenario.get("expected_cause", {})
    probable_causes = investigation.structured_report.get("causas_provaveis", [])
    top_cause = probable_causes[0] if probable_causes else {}
    expected_rank = _find_expected_rank(probable_causes, expected_cause)
    matched_terms = _find_matched_terms(investigation, scenario.get("expected_terms", []))
    unsupported_numbers = find_unsupported_numbers(investigation)

    return {
        "scenario_id": scenario["id"],
        "expected_cause": expected_cause,
        "top_cause": top_cause,
        "expected_rank": expected_rank,
        "top1_match": expected_rank == 1,
        "top_k": top_k,
        "top_k_match": expected_rank is not None and expected_rank <= top_k,
        "expected_terms": scenario.get("expected_terms", []),
        "matched_terms": matched_terms,
        "term_recall": _safe_ratio(len(matched_terms), len(scenario.get("expected_terms", []))),
        "unsupported_numbers": unsupported_numbers,
        "has_unsupported_numbers": bool(unsupported_numbers),
    }


def summarize_evaluations(evaluations):
    total = len(evaluations)
    top1_hits = sum(1 for item in evaluations if item["top1_match"])
    top_k_hits = sum(1 for item in evaluations if item["top_k_match"])
    unsupported = sum(1 for item in evaluations if item["has_unsupported_numbers"])
    execution_times = [
        item["tempo_execucao_segundos"]
        for item in evaluations
        if "tempo_execucao_segundos" in item
    ]

    return {
        "total_cenarios": total,
        "acertos_top1": top1_hits,
        "taxa_acerto_top1_percentual": _percentage(top1_hits, total),
        "acertos_top_k": top_k_hits,
        "top_k": evaluations[0]["top_k"] if evaluations else 0,
        "taxa_acerto_top_k_percentual": _percentage(top_k_hits, total),
        "cenarios_com_numeros_nao_suportados": unsupported,
        "taxa_numeros_nao_suportados_percentual": _percentage(unsupported, total),
        "tempo_medio_execucao_segundos": _average(execution_times),
    }


def build_evaluation_report(evaluations, summary):
    lines = [
        "# Relatorio de avaliacao dos cenarios",
        "",
        "## Resumo geral",
        "",
        f"- Cenarios avaliados: {summary['total_cenarios']}",
        (
            f"- Acerto Top 1: {summary['acertos_top1']} "
            f"({summary['taxa_acerto_top1_percentual']:.2f}%)"
        ),
        (
            f"- Acerto Top {summary['top_k']}: {summary['acertos_top_k']} "
            f"({summary['taxa_acerto_top_k_percentual']:.2f}%)"
        ),
        (
            "- Cenarios com numeros nao suportados pela base de evidencias: "
            f"{summary['cenarios_com_numeros_nao_suportados']}"
        ),
        (
            "- Tempo medio de execucao por cenario: "
            f"{summary['tempo_medio_execucao_segundos']:.4f} segundo(s)"
        ),
        "",
        "## Resultado por cenario",
        "",
        (
            "| Cenario | Causa esperada | Causa encontrada | Acerto Top 1 | "
            "Acerto Top K | Posicao esperada | Score encontrado | Tempo (s) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]

    for item in evaluations:
        top_cause = item["top_cause"]
        lines.append(
            "| "
            f"{item['scenario_id']} | "
            f"{_format_cause(item['expected_cause'])} | "
            f"{_format_cause(top_cause)} | "
            f"{_yes_no(item['top1_match'])} | "
            f"{_yes_no(item['top_k_match'])} | "
            f"{item['expected_rank'] or ''} | "
            f"{top_cause.get('score', '')} | "
            f"{item.get('tempo_execucao_segundos', '')} |"
        )

    lines.extend(
        [
            "",
            "## Observacoes",
            "",
            "- A avaliacao compara a causa esperada cadastrada em cada cenario com o ranking de causas provaveis produzido pelo agente.",
            "- A metrica Top 1 indica se a primeira causa ranqueada foi a causa esperada.",
            "- A metrica Top K indica se a causa esperada apareceu entre as primeiras causas ranqueadas.",
            "- A verificacao de numeros nao suportados procura valores numericos na conclusao que nao aparecem nas evidencias estruturadas.",
        ]
    )
    return "\n".join(lines) + "\n"


def find_unsupported_numbers(investigation, tolerance=0.01):
    conclusion = getattr(investigation, "final_conclusion", "")
    conclusion = _remove_period_codes(conclusion)
    numbers_in_conclusion = _extract_numbers(conclusion)
    allowed_numbers = []
    _collect_numbers(_evidence_dict(investigation), allowed_numbers)

    unsupported = []
    for raw_value, numeric_value in numbers_in_conclusion:
        if not _number_is_supported(numeric_value, allowed_numbers, tolerance):
            unsupported.append(raw_value)

    return _unique(unsupported)


def _find_expected_rank(probable_causes, expected_cause):
    if not expected_cause:
        return None

    for index, cause in enumerate(probable_causes, start=1):
        if _cause_matches(cause, expected_cause):
            return index
    return None


def _cause_matches(cause, expected_cause):
    return (
        _normalize(cause.get("dimensao", ""))
        == _normalize(expected_cause.get("dimensao", ""))
        and _normalize(cause.get("valor", "")) == _normalize(expected_cause.get("valor", ""))
    )


def _find_matched_terms(investigation, expected_terms):
    structured_report = dict(investigation.structured_report)
    structured_report.pop("anomalia", None)

    evidence = dict(investigation.evidence)
    evidence.pop("anomaly_description", None)

    source = json.dumps(
        {
            "structured_report": structured_report,
            "evidence": evidence,
            "final_conclusion": investigation.final_conclusion,
        },
        ensure_ascii=False,
    )
    normalized_source = _normalize(source)
    return [term for term in expected_terms if _normalize(term) in normalized_source]


def _evidence_dict(investigation):
    if hasattr(investigation, "to_dict"):
        data = investigation.to_dict()
    else:
        data = {
            "evidence": getattr(investigation, "evidence", {}),
            "structured_report": getattr(investigation, "structured_report", {}),
        }

    data = dict(data)
    data.pop("final_conclusion", None)
    return data


def _collect_numbers(value, output):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        output.append(float(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_numbers(item, output)
        return
    if isinstance(value, list):
        for item in value:
            _collect_numbers(item, output)


def _extract_numbers(text):
    matches = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    return [(match, float(match.replace(",", "."))) for match in matches]


def _remove_period_codes(text):
    return re.sub(r"\b\d{4}-\d{2}\b", "", text)


def _number_is_supported(number, allowed_numbers, tolerance):
    for allowed in allowed_numbers:
        if abs(number - allowed) <= tolerance:
            return True
        if abs(abs(number) - abs(allowed)) <= tolerance:
            return True
    return False


def _format_cause(cause):
    if not cause:
        return ""
    dimension = cause.get("dimensao", "")
    value = cause.get("valor", "")
    return f"{dimension}:{value}"


def _percentage(value, total):
    if total == 0:
        return 0.0
    return round((value / total) * 100, 2)


def _average(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _safe_ratio(value, total):
    if total == 0:
        return 0.0
    return round(value / total, 4)


def _normalize(value):
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _yes_no(value):
    return "sim" if value else "nao"
