import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src.data.contextual_demo import generate_contextual_sales
from src.data.loader import prepare_sales_data
from src.analysis.contextual_detection import analyze_contextual_series
from src.evaluation.contextual_benchmark import evaluate_contextual_dataset, run_contextual_benchmark, run_contextual_fixture


class ContextualBenchmarkTest(unittest.TestCase):
    def test_ground_truth_changes_metrics_but_not_predictions(self):
        dataset = generate_contextual_sales(seed=42)
        summary, first = evaluate_contextual_dataset(dataset.sales, dataset.calendar, dataset.truth)
        modified_truth = dataset.truth.copy(deep=True)
        modified_truth.loc[modified_truth["split"].eq("test"), "is_anomaly"] = False
        _, second = evaluate_contextual_dataset(dataset.sales, dataset.calendar, modified_truth)
        self.assertTrue(summary["available"])
        columns = ["data", "detector", "expected", "lower", "upper", "score", "flag", "eligible", "evaluation_mask"]
        pd.testing.assert_frame_equal(first[columns], second[columns])
        self.assertGreater(first.loc[first["evaluation_mask"], "is_anomaly"].sum(), 0)
        self.assertFalse(second["is_anomaly"].any())

    def test_shared_mask_and_conditional_context_counts(self):
        dataset = generate_contextual_sales(seed=137)
        summary, points = evaluate_contextual_dataset(dataset.sales, dataset.calendar, dataset.truth, seed=137)
        groups = {name: frame.reset_index(drop=True) for name, frame in points.groupby("detector")}
        pd.testing.assert_series_equal(groups["historical_without_context"]["evaluation_mask"], groups["contextual"]["evaluation_mask"])
        self.assertFalse(points.loc[points["split"].ne("test"), "evaluation_mask"].any())
        for row in summary["metrics"]:
            frame = groups[row["detector"]]
            expected = frame["evaluation_mask"] & frame["expected_context"] & ~frame["is_anomaly"]
            self.assertEqual(row["expected_context_points"], int(expected.sum()))
            self.assertEqual(row["expected_context_alerts"], int((expected & frame["flag"]).sum()))
            self.assertEqual(row["tp"] + row["fp"] + row["fn"] + row["tn"], row["evaluated_points"])

    def test_unavailable_context_does_not_publish_success(self):
        dataset = generate_contextual_sales()
        with patch("src.evaluation.contextual_benchmark.analyze_contextual_series", return_value={"available": False, "reason": "insufficient_context", "limitations": []}):
            summary, points = evaluate_contextual_dataset(dataset.sales, dataset.calendar, dataset.truth)
        self.assertFalse(summary["available"])
        self.assertEqual(summary["reason"], "insufficient_context")
        self.assertEqual(summary["metrics"], [])
        self.assertTrue(points.empty)

    def test_separate_reproducible_exports_and_aggregation(self):
        with TemporaryDirectory() as folder:
            first = run_contextual_benchmark(folder, seeds=[42, 137])
            point_bytes = Path(folder, "point_results.csv").read_bytes()
            second = run_contextual_benchmark(folder, seeds=[42, 137])
            self.assertEqual(first, second)
            self.assertEqual(point_bytes, Path(folder, "point_results.csv").read_bytes())
            json.dumps(second, allow_nan=False)
            self.assertEqual(first["available_runs"], 2)
            self.assertEqual(len(first["metrics"]), 4)
            calendar = pd.read_csv(Path(folder, "calendario.csv"))
            sales = pd.read_csv(Path(folder, "vendas.csv"))
            self.assertEqual(list(calendar.columns), ["data", "promocao"])
            self.assertNotIn("is_anomaly", sales)
            self.assertNotIn("is_anomaly", calendar)
            self.assertTrue(Path(folder, "gabarito.csv").is_file())
            for row in first["aggregate_metrics"]:
                self.assertEqual(row["runs"], 2)
                self.assertEqual(row["recall_n"], 2)
            self.assertIn("fictícios", Path(folder, "report.md").read_text(encoding="utf-8"))

    def test_seed_configuration_rejects_duplicates(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_contextual_benchmark(folder, seeds=[42, 42])

    def write_fixture(self, folder, dataset=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        dataset = dataset or generate_contextual_sales()
        for name, frame in (("vendas.csv", dataset.sales), ("calendario.csv", dataset.calendar), ("gabarito.csv", dataset.truth)):
            frame.to_csv(folder / name, index=False, date_format="%Y-%m-%d")
        return folder

    def test_fixture_reads_actual_files_without_generator_or_label_leakage(self):
        with TemporaryDirectory() as folder:
            source = self.write_fixture(Path(folder, "inputs"))
            sales = pd.read_csv(source / "vendas.csv")
            product = sales["produto"].eq("Notebook")
            sales.loc[product, ["preco_unitario", "faturamento"]] *= 1.02
            sales["is_anomaly"] = True
            sales.to_csv(source / "vendas.csv", index=False)
            truth = pd.read_csv(source / "gabarito.csv")
            truth["revenue"] = truth["data"].map(sales.groupby("data")["faturamento"].sum())
            truth.to_csv(source / "gabarito.csv", index=False)
            calendar = pd.read_csv(source / "calendario.csv")
            calendar.loc[len(calendar) - 1, "promocao"] = True
            calendar["answer_key"] = True
            calendar.to_csv(source / "calendario.csv", index=False)
            with patch("src.evaluation.contextual_benchmark.generate_contextual_sales", side_effect=AssertionError("No regenerated fixture")), \
                 patch("src.evaluation.contextual_benchmark.analyze_contextual_series", wraps=analyze_contextual_series) as detector:
                result = run_contextual_fixture(source, Path(folder, "results"))
            self.assertEqual(result["mode"], "offline_synthetic_fixture")
            self.assertEqual(result["seeds"], [42])
            self.assertFalse(result["source"]["regenerated"])
            self.assertTrue(result["source"]["truth_revenue_reconciled"])
            self.assertEqual(result["source"]["files"]["vendas.csv"]["sha256"], hashlib.sha256((source / "vendas.csv").read_bytes()).hexdigest())
            self.assertEqual(set(detector.call_args.args[1].columns), {"data", "promocao"})
            self.assertNotIn("is_anomaly", detector.call_args.args[0])
            exported = pd.read_csv(result["outputs"]["vendas.csv"])
            expected = prepare_sales_data(sales)
            pd.testing.assert_frame_equal(prepare_sales_data(exported), expected)
            exported_calendar = pd.read_csv(result["outputs"]["calendario.csv"])
            pd.testing.assert_frame_equal(exported_calendar, calendar[["data", "promocao"]])
            points = pd.read_csv(result["outputs"]["point_results.csv"])
            observed = points.groupby("data")["valor"].first()
            pd.testing.assert_series_equal(observed, sales.groupby("data")["faturamento"].sum().rename("valor"))

    def test_fixture_rejects_stale_revenue_answer_key(self):
        with TemporaryDirectory() as folder:
            source = self.write_fixture(Path(folder, "inputs"))
            sales = pd.read_csv(source / "vendas.csv")
            sales.loc[0, "faturamento"] *= 2
            sales.to_csv(source / "vendas.csv", index=False)
            destination = Path(folder, "results")
            with self.assertRaisesRegex(ValueError, "gabarito"):
                run_contextual_fixture(source, destination)
            self.assertFalse((destination / "summary.json").exists())

    def test_fixture_preserves_unknown_calendar_date(self):
        with TemporaryDirectory() as folder:
            source = self.write_fixture(Path(folder, "inputs"))
            calendar = pd.read_csv(source / "calendario.csv")
            missing_date = calendar.iloc[-1]["data"]
            calendar.iloc[:-1].to_csv(source / "calendario.csv", index=False)
            result = run_contextual_fixture(source, Path(folder, "results"))
            points = pd.read_csv(result["outputs"]["point_results.csv"])
            contextual = points.loc[points["detector"].eq("contextual") & points["data"].eq(missing_date)].iloc[0]
            self.assertFalse(contextual["context_known"])
            self.assertFalse(contextual["eligible"])
            self.assertFalse(contextual["evaluation_mask"])
            self.assertTrue(pd.isna(contextual["promocao"]))
            self.assertEqual(contextual["context_label"], "unknown")
            self.assertEqual(len(pd.read_csv(result["outputs"]["calendario.csv"])), 419)
            self.assertEqual(result["runs"][0]["eligible_test_points"], 104)

    def test_fixture_cannot_overwrite_its_input_directory(self):
        with TemporaryDirectory() as folder:
            source = self.write_fixture(folder)
            before = {path.name: path.read_bytes() for path in source.glob("*.csv")}
            with self.assertRaisesRegex(ValueError, "diferente"):
                run_contextual_fixture(source, source)
            self.assertEqual(before, {path.name: path.read_bytes() for path in source.glob("*.csv")})


if __name__ == "__main__":
    unittest.main()
