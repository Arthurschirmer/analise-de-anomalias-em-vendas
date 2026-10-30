import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd
from src.evaluation.anomaly_benchmark import (
    _inject, controlled_sales, evaluate_stage, run_anomaly_benchmark)
from src.analysis.historical_detection import build_time_series, temporal_split


class BenchmarkTest(unittest.TestCase):
    def test_legacy_volume_injection_allocates_total_for_unit_rows(self):
        data = controlled_sales(days=180)
        data = data.loc[data.index.repeat(5)].reset_index(drop=True)
        data["quantidade"] = 1
        data["faturamento"] = data["preco_unitario"]
        changed, events, labels = _inject(data, temporal_split(build_time_series(data)))
        self.assertEqual(events[0]["status"], "completed")
        for day in events[0]["daily_effects"]:
            self.assertEqual(day["original_quantity"], 20)
            self.assertEqual(day["target_quantity"], 5)
            self.assertEqual(day["realized_quantity"], 5)
            self.assertIn(pd.Timestamp(day["data"]), labels)
        self.assertTrue(((changed["quantidade"] * changed["preco_unitario"]
                          - changed["faturamento"]).abs() < 1e-8).all())

    def test_reproducible_labels_shared_mask_and_strict_exports(self):
        with TemporaryDirectory() as folder:
            one = run_anomaly_benchmark(folder, seed=23, days=180)
            first = Path(folder, "point_results.csv").read_bytes()
            two = run_anomaly_benchmark(folder, seed=23, days=180)
            self.assertEqual(one, two)
            self.assertEqual(first, Path(folder, "point_results.csv").read_bytes())
            json.dumps(two, allow_nan=False)
            stage = two["stages"]["controlled"]
            self.assertEqual({item["detector"] for item in stage["metrics"]},
                             {"previous_week_30pct", "historical_quantile98", "seasonal_trend_mad"})
            self.assertEqual(stage["calibration"]["method_key"], "robust")
            self.assertEqual(stage["legacy_calibration"]["method_key"], "quantile")
            points = pd.read_csv(Path(folder, "point_results.csv"))
            before = points.loc[points["detector"].eq("previous_week_30pct"), "evaluation_mask"]
            after = points.loc[points["detector"].eq("seasonal_trend_mad"), "evaluation_mask"]
            self.assertEqual(before.tolist(), after.tolist())
            self.assertFalse(points.loc[points["split"].ne("test"), "is_anomaly"].any())

    def test_real_uninjected_points_are_not_normal_or_false_positives(self):
        summary, points = evaluate_stage(controlled_sales(days=180), "real", False)
        unknown = points.loc[points["label_status"].eq("unlabeled_real")]
        self.assertTrue(unknown["is_anomaly"].isna().all())
        for metrics in summary["metrics"]:
            for field in ["precision", "recall", "f1", "fp", "tn"]:
                self.assertIsNone(metrics[field])
            self.assertGreater(metrics["known_positive_points"], 0)

    def test_injections_preserve_input_and_sales_math(self):
        data = controlled_sales(days=180)
        before = data.copy(deep=True)
        series = temporal_split(build_time_series(data))
        changed, events, labels = _inject(data, series)
        pd.testing.assert_frame_equal(before, data)
        self.assertTrue(((changed["quantidade"] * changed["preco_unitario"]
                          - changed["faturamento"]).abs() < 1e-8).all())
        self.assertEqual(len(events), 4)
        test_start = series.loc[series["split"].eq("test"), "data"].min()
        self.assertTrue(all(date >= test_start for date in labels))
        pd.testing.assert_frame_equal(changed.loc[changed["data"] < test_start].astype(
            {"quantidade": float, "preco_unitario": float, "faturamento": float}),
            before.loc[before["data"] < test_start].astype(
            {"quantidade": float, "preco_unitario": float, "faturamento": float}))

