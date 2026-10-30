import csv
import json
import time
from pathlib import Path

from src.agent.investigation_agent import InvestigationAgent
from src.data.adapters import load_sales_dataset
from src.evaluation.agent_evaluator import (
    build_evaluation_report,
    evaluate_investigation,
    summarize_evaluations,
)
from src.llm.factory import create_llm_provider_from_env


def load_scenarios(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_scenarios(scenario_file, output_dir):
    scenarios = load_scenarios(scenario_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    agent = InvestigationAgent(llm_provider=create_llm_provider_from_env())
    summary = []
    evaluations = []

    for scenario in scenarios:
        data = _load_scenario_data(scenario)
        start_time = time.perf_counter()
        investigation = agent.investigate(scenario["anomalia"], data)
        evaluation = evaluate_investigation(investigation, scenario)
        evaluation["tempo_execucao_segundos"] = round(
            time.perf_counter() - start_time,
            4,
        )
        investigation.evaluation = evaluation
        evaluations.append(evaluation)

        log_path = output_path / f"{scenario['id']}.json"
        investigation.save(log_path)

        top_cause = evaluation["top_cause"]
        expected_cause = evaluation["expected_cause"]

        summary.append(
            {
                "id": scenario["id"],
                "anomalia": scenario["anomalia"],
                "log": str(log_path),
                "causa_esperada_dimensao": expected_cause.get("dimensao", ""),
                "causa_esperada_valor": expected_cause.get("valor", ""),
                "causa_encontrada_dimensao": top_cause.get("dimensao", ""),
                "causa_encontrada_valor": top_cause.get("valor", ""),
                "score_causa_encontrada": top_cause.get("score", ""),
                "confianca_causa_encontrada": top_cause.get("confianca", ""),
                "posicao_causa_esperada": evaluation["expected_rank"] or "",
                "acerto_top1": _yes_no(evaluation["top1_match"]),
                "acerto_top_k": _yes_no(evaluation["top_k_match"]),
                "termos_esperados": "; ".join(evaluation["expected_terms"]),
                "termos_encontrados": "; ".join(evaluation["matched_terms"]),
                "qtd_termos_encontrados": len(evaluation["matched_terms"]),
                "numeros_nao_suportados": "; ".join(evaluation["unsupported_numbers"]),
                "tempo_execucao_segundos": evaluation["tempo_execucao_segundos"],
            }
        )

    summary_path = output_path / "summary.csv"
    _save_summary(summary_path, summary)
    evaluation_summary = summarize_evaluations(evaluations)
    _save_json(output_path / "evaluation_summary.json", evaluation_summary)
    _save_text(
        output_path / "evaluation_report.md",
        build_evaluation_report(evaluations, evaluation_summary),
    )
    return summary_path, summary


def _load_scenario_data(scenario):
    dataset_type = scenario.get("dataset_type", "csv")
    return load_sales_dataset(
        dataset_type=dataset_type,
        csv_path=scenario.get("csv"),
        input_dir=scenario.get("input_dir"),
    )


def _save_summary(path, rows):
    if not rows:
        return

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _save_text(path, text):
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def _yes_no(value):
    return "sim" if value else "nao"
