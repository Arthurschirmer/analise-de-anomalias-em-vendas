"""Side-by-side explanations over identical evidence, with human review pending."""
import hashlib
import json
import math
import re
import time
from pathlib import Path

from src.agent.investigation_agent import conclusion_request
from src.evaluation.anomaly_benchmark import clean_json
from src.llm.mock_provider import MockLLMProvider
from src.ui.feedback import classify_error, sanitize_details


def _numeric_evidence(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _numeric_evidence(item, path + "." + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _numeric_evidence(item, path + f"[{i}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield path, float(value)
        if "desconto" in path:
            yield path + " (em pontos percentuais)", float(value) * 100


def numeric_correspondence(text, evidence):
    """Lexical aid only: does not validate meaning, derivations or causal claims."""
    evidence_values = list(_numeric_evidence(clean_json(evidence)))
    scrubbed = re.sub(r"\b\d{4}-\d{2}(?:-\d{2})?\b", " ", text)
    scrubbed = re.sub(r"(?m)^\s*\d+[.)]\s+", "", scrubbed)
    scrubbed = re.sub(r"\b[a-zA-Z0-9_-]{12,}\b", " ", scrubbed)
    pattern = r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?![\w])"
    checks = []
    for match in re.finditer(pattern, scrubbed):
        token = match.group()
        raw = token
        if "," in raw and "." in raw:
            decimal = "," if raw.rfind(",") > raw.rfind(".") else "."
            raw = raw.replace("." if decimal == "," else ",", "").replace(decimal, ".")
        elif "," in raw:
            # A comma is treated as decimal in Portuguese; ambiguous forms are
            # retained as a heuristic, never a confirmed factual error.
            raw = raw.replace(",", ".")
        elif raw.count(".") > 1:
            raw = raw.replace(".", "")
        try:
            number = float(raw)
        except ValueError:
            continue
        decimals = len(raw.rsplit(".", 1)[1]) if "." in raw else 0
        tolerance = .5 * 10 ** (-decimals) + 1e-8
        paths = [path for path, value in evidence_values if abs(number - value) <= tolerance]
        checks.append({"token": token, "value": number, "matched": bool(paths),
                       "evidence_paths": paths[:3]})
    return {"method": "heuristic_numeric_correspondence", "numbers": checks,
            "matched": sum(item["matched"] for item in checks), "total": len(checks),
            "needs_review": [item["token"] for item in checks if not item["matched"]],
            "limitation": "Correspondência numérica não verifica contexto, unidades, causalidade ou cálculos derivados; revise manualmente."}


def compare_investigation_reports(investigation, provider=None):
    system, user, context = conclusion_request(
        investigation.anomaly_description, investigation.evidence, investigation.structured_report)
    serialized = json.dumps(clean_json(context), ensure_ascii=False, sort_keys=True, allow_nan=False)
    prompt_hash = hashlib.sha256((system + user + serialized).encode("utf-8")).hexdigest()
    result = {"version": 1, "evidence_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
              "request_sha256": prompt_hash, "context": clean_json(context),
              "system_prompt": system, "user_prompt": user,
              "human_review": {"status": "pending", "local": {}, "ai": {}, "notes": ""},
              "rubric": {"fidelity": "1: contradiz evidências; 5: fiel aos números e limitações.",
                         "clarity": "1: difícil de entender; 5: clara para o público definido.",
                         "usefulness": "1: sem orientação útil; 5: prioriza verificações fundamentadas."}}
    started = time.perf_counter()
    local = MockLLMProvider().generate(system, user, context)
    result["local"] = {"status": "completed", "provider": "deterministic_local",
                       "text": local, "latency_seconds": time.perf_counter() - started,
                       "numeric_check": numeric_correspondence(local, context), "token_usage": None}
    result["ai"] = {"status": "not_run", "model": None, "text": None, "token_usage": None,
                    "latency_seconds": None}
    if provider is not None and not isinstance(provider, MockLLMProvider):
        started = time.perf_counter()
        result["ai"]["model"] = getattr(provider, "model", type(provider).__name__)
        try:
            text = provider.generate(system, user, clean_json(context))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("A IA retornou uma resposta vazia.")
            result["ai"].update(status="completed", text=text,
                                numeric_check=numeric_correspondence(text, context),
                                token_usage=getattr(provider, "last_usage", None))
        except Exception as error:
            feedback = classify_error(error)
            result["ai"].update(status="failed", error_code=feedback["code"],
                                error=sanitize_details(str(error)))
        result["ai"]["latency_seconds"] = time.perf_counter() - started
    return clean_json(result)


def save_comparison(result, output_dir="outputs/model_comparison"):
    out = Path(output_dir) / result["request_sha256"][:16]
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Comparação de explicações", "", "Evidências SHA-256: " + result["evidence_sha256"],
             "", "## Relatório local", result["local"]["text"], "", "## Relatório da IA",
             result["ai"].get("text") or "Estado: " + result["ai"]["status"],
             "", "## Avaliação humana", json.dumps(result["human_review"], ensure_ascii=False),
             "", "A verificação numérica é heurística. Latência de uma chamada não representa desempenho geral."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out

