import unittest
import numpy as np
import pandas as pd

from src.analysis.historical_detection import (
    apply_threshold, build_time_series, calibrate_threshold, score_historical_series, temporal_split)
from src.evaluation.anomaly_benchmark import controlled_sales


class HistoricalDetectionTest(unittest.TestCase):
    def frame(self, values):
        return build_time_series(pd.DataFrame({"data": pd.date_range("2024-01-01", periods=len(values)),
                                              "faturamento": values}))

    def test_current_and_future_values_cannot_change_past_reference(self):
        frame = self.frame(np.arange(140, dtype=float) + 100)
        before = score_historical_series(frame, trend="linear")
        altered = frame.copy()
        altered.loc[100:, "valor"] *= 20
        after = score_historical_series(altered, trend="linear")
        pd.testing.assert_frame_equal(before.iloc[:100], after.iloc[:100])
        self.assertEqual(before.loc[100, "expected"], after.loc[100, "expected"])
        self.assertTrue(after.loc[100, "flag"])

    def test_weekday_and_linear_trend_are_not_anomalies(self):
        values = np.array([100 + (i % 7) * 50 + i * 2 for i in range(140)], dtype=float)
        frame = score_historical_series(self.frame(values), trend="linear")
        eligible = frame.loc[frame["eligible"]]
        self.assertTrue(np.allclose(eligible["valor"], eligible["expected"]))
        self.assertFalse(eligible["flag"].any())

    def test_zero_mad_is_finite_and_spike_detected(self):
        frame = self.frame([100.] * 70 + [180.])
        result = score_historical_series(frame)
        self.assertTrue(np.isfinite(result.iloc[-1]["score"]))
        self.assertEqual(result.iloc[-1]["scale"], 5)
        self.assertTrue(result.iloc[-1]["flag"])

    def test_unknown_dates_and_partial_weeks_do_not_become_zero(self):
        raw = pd.DataFrame({"data": pd.date_range("2024-01-03", periods=30), "faturamento": 100.})
        raw = raw.drop(index=10)
        daily = build_time_series(raw)
        self.assertTrue(pd.isna(daily.iloc[10]["valor"]))
        weekly = build_time_series(raw, frequency="W")
        self.assertFalse(weekly.iloc[0]["completo"])
        self.assertFalse(weekly.iloc[-1]["completo"])
        self.assertTrue(weekly["valor"].isna().any())
        filled = build_time_series(raw, missing_policy="zero")
        self.assertEqual(filled.iloc[10]["valor"], 0)
        self.assertFalse(filled.iloc[10]["observado"])

    def test_calibration_ignores_test_scores_and_labels(self):
        frame = temporal_split(build_time_series(controlled_sales()))
        scored = score_historical_series(frame, trend="linear")
        threshold = calibrate_threshold(scored)
        altered = scored.copy()
        altered.loc[altered["split"].eq("test"), "score"] = 1e10
        self.assertEqual(threshold, calibrate_threshold(altered))
        self.assertTrue(threshold["frozen"])
        self.assertEqual(apply_threshold(scored, threshold).attrs["calibration"], threshold)

    def test_short_history_and_interleaved_split_rejected(self):
        with self.assertRaises(ValueError):
            temporal_split(self.frame([1.] * 30))
        frame = score_historical_series(temporal_split(self.frame([100.] * 140)))
        frame.loc[30, "split"] = "test"
        with self.assertRaises(ValueError):
            calibrate_threshold(frame)

    def test_robust_calibration_resists_isolated_extreme_score(self):
        frame = score_historical_series(temporal_split(self.frame([100.] * 140)))
        calibration = frame["split"].eq("calibration")
        frame.loc[calibration, "score"] = 1.0
        frame.loc[calibration, "eligible"] = True
        frame.loc[frame.index[calibration][-1], "score"] = 100.0
        robust = calibrate_threshold(frame, method="robust")
        quantile = calibrate_threshold(frame, method="quantile")
        self.assertEqual(robust["threshold"], 3.0)
        self.assertEqual(quantile["threshold"], 100.0)
        self.assertTrue(robust["diagnostics"]["tail_sensitive"])
        self.assertEqual(robust["diagnostics"]["calibration_exceedances"], 1)
        self.assertEqual(robust["method_key"], "robust")

    def test_robust_calibration_never_uses_test_scores_or_labels(self):
        frame = score_historical_series(temporal_split(self.frame([100.] * 140)))
        before = calibrate_threshold(frame, method="robust")
        frame.loc[frame["split"].eq("test"), "score"] = 1e12
        frame["is_anomaly"] = frame["split"].eq("test")
        self.assertEqual(before, calibrate_threshold(frame, method="robust"))

    def test_calibration_method_and_multiplier_validated(self):
        frame = score_historical_series(temporal_split(self.frame([100.] * 140)))
        with self.assertRaises(ValueError):
            calibrate_threshold(frame, method="unsupported")
        with self.assertRaises(ValueError):
            calibrate_threshold(frame, method="robust", robust_multiplier=float("nan"))

