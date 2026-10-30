import argparse
import json
from pathlib import Path

from src.evaluation.scenario_runner import run_scenarios


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa cenarios experimentais de investigacao de anomalias."
    )
    parser.add_argument(
        "--scenarios",
        default="data/scenarios.json",
        help="Arquivo JSON com os cenarios experimentais.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments",
        help="Pasta onde os logs dos cenarios serao salvos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path, summary = run_scenarios(args.scenarios, args.output_dir)
    evaluation_summary_path = Path(args.output_dir) / "evaluation_summary.json"
    evaluation_report_path = Path(args.output_dir) / "evaluation_report.md"

    print(f"Cenarios executados: {len(summary)}")
    print(f"Resumo salvo em: {summary_path}")

    if evaluation_summary_path.exists():
        with open(evaluation_summary_path, "r", encoding="utf-8") as file:
            evaluation_summary = json.load(file)
        print(
            "Taxa de acerto Top 1: "
            f"{evaluation_summary['taxa_acerto_top1_percentual']:.2f}%"
        )
        print(
            "Taxa de acerto Top K: "
            f"{evaluation_summary['taxa_acerto_top_k_percentual']:.2f}%"
        )
        print(
            "Tempo medio por cenario: "
            f"{evaluation_summary['tempo_medio_execucao_segundos']:.4f}s"
        )
        print(f"Relatorio de avaliacao salvo em: {evaluation_report_path}")


if __name__ == "__main__":
    main()
