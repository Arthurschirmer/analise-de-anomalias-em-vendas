"""Run the offline, reproducible detector benchmark."""
import argparse
from src.evaluation.anomaly_benchmark import run_anomaly_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=420)
    parser.add_argument("--olist-dir")
    parser.add_argument("--suite", action="store_true", help="Executar suíte com cenários graduais, persistentes e compensados.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2026])
    args = parser.parse_args()
    if args.suite:
        from src.evaluation.benchmark_suite import run_benchmark_suite
        result = run_benchmark_suite(output_dir=args.output_dir, seeds=args.seeds,
                                     days=args.days, olist_dir=args.olist_dir)
        print("Execuções avaliadas:", len(result["runs"]))
        print("Sementes:", result["seeds"])
    else:
        result = run_anomaly_benchmark(output_dir=args.output_dir, seed=args.seed,
                                       days=args.days, olist_dir=args.olist_dir)
        for name, stage in result["stages"].items():
            print(name, stage["metrics"])
    print("Relatório:", result["outputs"]["report.md"])

