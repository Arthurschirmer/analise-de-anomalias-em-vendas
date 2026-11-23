"""Executa o experimento fictício de promoções e quedas inesperadas."""
import argparse

from src.evaluation.contextual_benchmark import run_contextual_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/contextual")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2026])
    parser.add_argument("--days", type=int, default=420)
    args = parser.parse_args()
    result = run_contextual_benchmark(output_dir=args.output_dir, seeds=args.seeds, days=args.days)
    print("Comparações disponíveis:", result["available_runs"], "de", len(result["runs"]))
    print("Relatório:", result["outputs"]["report.md"])
