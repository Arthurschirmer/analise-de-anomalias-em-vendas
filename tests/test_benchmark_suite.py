import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.anomaly_benchmark import controlled_sales
from src.evaluation.benchmark_suite import (
    SCENARIOS, _inject_scenario, _split_original, run_benchmark_suite)


class BenchmarkSuiteTest(unittest.TestCase):
    def test_unit_rows_receive_mild_volume_changes_at_daily_total(self):
        data = controlled_sales(days=180)
        data = data.loc[data.index.repeat(5)].reset_index(drop=True)
        data["quantidade"] = 1
        data["faturamento"] = data["preco_unitario"]
        original = data.copy(deep=True)
        series = _split_original(data)
        for name, expected in [("mild_drop", 18), ("mild_spike", 22),
                               ("moderate_drop", 14), ("moderate_spike", 26),
                               ("strong_drop", 8), ("strong_spike", 32)]:
            spec = next(item for item in SCENARIOS if item["name"] == name)
            changed, event, positives = _inject_scenario(data, series, spec, 19)
            self.assertEqual(event["status"], "completed")
            self.assertEqual(len(positives), 2)
            self.assertTrue((changed["quantidade"] % 1 == 0).all())
            self.assertTrue(np.allclose(changed["preco_unitario"], data["preco_unitario"]))
            self.assertTrue(np.allclose(changed["quantidade"] * changed["preco_unitario"], changed["faturamento"]))
            for day in event["daily_effects"]:
                self.assertEqual(day["original_quantity"], 20)
                self.assertEqual(day["target_quantity"], expected)
                self.assertEqual(day["realized_quantity"], expected)
                self.assertNotEqual(day["target_revenue_change_pct"], 0)
            repeated, _, _ = _inject_scenario(data, series, spec, 19)
            pd.testing.assert_frame_equal(changed, repeated)
        pd.testing.assert_frame_equal(data, original)

    def test_rounding_no_effect_is_reported_without_positive_labels(self):
        data = controlled_sales(days=180)
        data["quantidade"] = 1
        data["faturamento"] = data["preco_unitario"]
        series = _split_original(data)
        changed, event, positives = _inject_scenario(data, series,
            {"name": "tiny_drop", "type": "volume_drop", "intensity": .001, "duration": 2}, 7)
        self.assertEqual(event["status"], "no_effect")
        self.assertEqual(positives, set())
        self.assertTrue(np.allclose(changed["faturamento"], data["faturamento"]))
        self.assertTrue(all(day["target_quantity"] == day["original_quantity"] for day in event["daily_effects"]))

    def test_seeded_suite_exports_strict_metrics_without_network(self):
        with TemporaryDirectory() as folder, patch("urllib.request.urlopen", side_effect=AssertionError("offline")):
            settings = dict(seeds=(7, 8), days=180, scenario_names=["mild_drop", "gradual_drop"])
            one = run_benchmark_suite(folder, **settings)
            first = Path(folder, "suite_point_results.csv").read_bytes()
            two = run_benchmark_suite(folder, **settings)
            self.assertEqual(one, two)
            self.assertEqual(first, Path(folder, "suite_point_results.csv").read_bytes())
            json.dumps(two, allow_nan=False)
            self.assertTrue(Path(two["outputs"]["suite_injection_effects.csv"]).is_file())
            self.assertTrue(two["injection_effects"])
            self.assertEqual(len(two["runs"]), 12)
            self.assertTrue(all(item["runs"] == 2 for item in two["aggregate_metrics"]))
            self.assertTrue(all(item["injection_recall_n"] == 2 for item in two["aggregate_metrics"]))
            points = pd.read_csv(Path(folder, "suite_point_results.csv"))
            self.assertFalse(points.loc[points["split"].ne("test"), "is_anomaly"].any())
            for _, group in points.groupby(["seed", "scenario", "scope"]):
                masks = [part["evaluation_mask"].tolist() for _, part in group.groupby("detector")]
                self.assertTrue(all(mask == masks[0] for mask in masks))
            for manifest in two["manifests"]:
                self.assertLess(manifest["splits"]["calibration"]["end"],
                                manifest["splits"]["test"]["start"])

    def test_compensated_segment_changes_are_hidden_in_total_and_preserve_math(self):
        data = controlled_sales(days=180)
        original = data.copy(deep=True)
        series = _split_original(data)
        spec = next(item for item in SCENARIOS if item["name"] == "category_hidden_drop")
        changed, event, positive = _inject_scenario(data, series, spec, 7)
        pd.testing.assert_frame_equal(data, original)
        self.assertEqual(len(positive), 5)
        self.assertTrue(event["total_revenue_preserved"])
        before = data.groupby("data")["faturamento"].sum()
        after = changed.groupby("data")["faturamento"].sum()
        self.assertTrue(np.allclose(before, after))
        self.assertTrue(np.allclose(changed["quantidade"] * changed["preco_unitario"], changed["faturamento"]))
        train_end = series.loc[series["split"].eq("train"), "data"].max()
        expected_target = str(data.loc[data["data"].le(train_end)].groupby("categoria")["faturamento"].sum().idxmax())
        self.assertEqual(event["target"], expected_target)
        self.assertTrue(all(date > train_end for date in positive))
        with TemporaryDirectory() as folder:
            result = run_benchmark_suite(folder, seeds=(7,), days=180,
                                         scenario_names=["category_hidden_drop"])
            global_rows = [item for item in result["runs"] if item["scope"] == "global"]
            segment_rows = [item for item in result["runs"] if item["scope"] != "global"]
            self.assertTrue(all(item["known_positive_points"] == 0 for item in global_rows))
            self.assertTrue(all(item["f1"] is None and item["precision"] is None
                                and item["recall"] is None for item in global_rows))
            self.assertTrue(all(item["false_alert_rate"] is not None for item in global_rows))
            self.assertTrue(all(item["known_positive_points"] == 5 for item in segment_rows))
            self.assertTrue(all(item["recall_mean"] is None for item in result["aggregate_metrics"]
                                if item["scope"] == "global"))
            self.assertTrue(all(item["f1_n"] == 0 and item["injection_recall_n"] == 0
                                for item in result["aggregate_metrics"] if item["scope"] == "global"))

    def test_normal_control_reports_false_alerts_without_positive_recovery_metrics(self):
        with TemporaryDirectory() as folder:
            result = run_benchmark_suite(folder, seeds=(7,), days=180,
                                         scenario_names=["normal_control"])
            for item in result["runs"]:
                self.assertEqual(item["metrics_applicability"], "no_evaluable_positives")
                self.assertEqual(item["known_positive_points"], 0)
                for key in ["precision", "recall", "f1", "injection_recall"]:
                    self.assertIsNone(item[key])
                self.assertAlmostEqual(item["false_alert_rate"], item["fp"] / (item["fp"] + item["tn"]))
            for item in result["aggregate_metrics"]:
                self.assertEqual(item["f1_n"], 0)
                self.assertEqual(item["false_alert_rate_n"], 1)

    def test_real_split_is_calendar_declared_and_unknown_labels_stay_unknown(self):
        data = controlled_sales(days=577)
        data["data"] -= pd.Timestamp("2023-01-02") - pd.Timestamp("2017-01-01")
        split = _split_original(data, real=True)
        self.assertEqual(split.loc[split["data"].eq("2017-08-31"), "split"].iloc[0], "train")
        self.assertEqual(split.loc[split["data"].eq("2017-09-01"), "split"].iloc[0], "calibration")
        self.assertEqual(split.loc[split["data"].eq("2018-01-01"), "split"].iloc[0], "test")
        with TemporaryDirectory() as folder:
            result = run_benchmark_suite(folder, seeds=(7,), days=180, real_data=data,
                                         scenario_names=["mild_drop"])
            metrics = [item for item in result["runs"] if item["stage"] == "olist_semisynthetic"]
            self.assertEqual(len(metrics), 3)
            for item in metrics:
                for field in ["precision", "recall", "f1", "fp", "tn"]:
                    self.assertIsNone(item[field])
            points = pd.read_csv(Path(folder, "suite_point_results.csv"))
            unknown = points.loc[points["stage"].eq("olist_semisynthetic")
                                 & points["label_status"].eq("unlabeled_real")]
            self.assertTrue(unknown["is_anomaly"].isna().all())
            manifest = next(item for item in result["manifests"] if item["stage"] == "olist_semisynthetic")
            self.assertEqual(manifest["evaluation_status"], "exploratory_previously_viewed_data")

    def test_invalid_scenarios_and_seeds_are_rejected(self):
        with TemporaryDirectory() as folder:
            for seeds in [(), (7, 7), (-1,)]:
                with self.assertRaises(ValueError):
                    run_benchmark_suite(folder, seeds=seeds)
            with self.assertRaises(ValueError):
                run_benchmark_suite(folder, seeds=(7,), scenario_names=["unknown"])

