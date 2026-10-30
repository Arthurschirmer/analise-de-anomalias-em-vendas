import unittest

import pandas as pd

from src.analysis.historical_detection import (
    apply_threshold, build_time_series, calibrate_threshold, score_historical_series, temporal_split)
from src.analysis.segment_detection import detect_segment_anomalies


def sales(days=168, start="2024-01-01"):
    return pd.DataFrame([
        {"data": date, "categoria": group, "regiao": "SP" if group == "A" else "PR",
         "faturamento": 50.0, "quantidade": 5.0}
        for date in pd.date_range(start, periods=days) for group in ["A", "B"]])


class SegmentDetectionTest(unittest.TestCase):
    def test_hidden_category_drop_detected_while_total_stays_constant(self):
        data = sales()
        last = data["data"].max()
        data.loc[data["data"].eq(last) & data["categoria"].eq("A"), "faturamento"] = 10.
        data.loc[data["data"].eq(last) & data["categoria"].eq("B"), "faturamento"] = 90.
        global_points = score_historical_series(temporal_split(build_time_series(data)), trend="linear")
        global_points = apply_threshold(global_points, calibrate_threshold(global_points, method="robust"))
        self.assertFalse(global_points.iloc[-1]["flag"])
        result = detect_segment_anomalies(data)
        flagged = result["points"].loc[result["points"]["data"].eq(last)]
        self.assertTrue(flagged["flag"].all())
        self.assertTrue(result["summary"]["status"].eq("evaluated").all())
        self.assertEqual(result["metadata"]["selection_policy"], "training_support_only")
        self.assertEqual(set(result["metadata"]["calibrations"]), {"A", "B"})
        self.assertEqual(result["metadata"]["split_boundaries"], global_points.attrs["split_boundaries"])

    def test_state_detection_uses_uf_column(self):
        result = detect_segment_anomalies(sales(), dimension="regiao")
        self.assertEqual(set(result["points"]["segmento"]), {"SP", "PR"})
        self.assertTrue(result["points"]["dimensao"].eq("regiao").all())

    def test_top_selection_does_not_use_test_revenue(self):
        data = sales()
        before = detect_segment_anomalies(data, max_segments=1)
        altered = data.copy()
        altered.loc[altered["data"].ge(pd.Timestamp("2024-05-15")) & altered["categoria"].eq("B"),
                    "faturamento"] = 1e9
        after = detect_segment_anomalies(altered, max_segments=1)
        self.assertEqual(before["metadata"]["selected_segments"], ["A"])
        self.assertEqual(after["metadata"]["selected_segments"], ["A"])
        self.assertEqual(before["metadata"]["calibrations"], after["metadata"]["calibrations"])
        self.assertEqual(after["summary"].set_index("segmento").loc["B", "reason"],
                         "outside_training_top_limit")

    def test_new_and_sparse_groups_are_skipped_explicitly(self):
        data = sales()
        sparse = data.loc[data.index % 28 == 0].copy()
        sparse["categoria"] = "Rare"
        future = data.loc[data["data"].eq(data["data"].max())].iloc[:1].copy()
        future["categoria"] = "New"
        result = detect_segment_anomalies(pd.concat([data, sparse, future], ignore_index=True))
        summary = result["summary"].set_index("segmento")
        self.assertEqual(summary.loc["Rare", "reason"], "insufficient_training_observations")
        self.assertEqual(summary.loc["New", "reason"], "not_observed_in_training")
        self.assertEqual(summary.loc["Rare", "status"], "skipped")
        self.assertNotIn("Rare", set(result["points"]["segmento"]))

    def test_missing_segment_day_is_unknown_by_default(self):
        data = sales()
        day = pd.Timestamp("2024-05-15")
        data = data.loc[~(data["categoria"].eq("A") & data["data"].eq(day))]
        result = detect_segment_anomalies(data)
        point = result["points"].loc[result["points"]["segmento"].eq("A")
                                    & result["points"]["data"].eq(day)].iloc[0]
        self.assertTrue(pd.isna(point["valor"]))
        self.assertFalse(point["eligible"])
        self.assertFalse(point["flag"])

    def test_explicit_zero_never_fills_global_gaps_or_before_first_activity(self):
        data = sales()
        gap = pd.Timestamp("2024-05-15")
        debut = pd.Timestamp("2024-01-15")
        segment_gap = pd.Timestamp("2024-05-16")
        data = data.loc[~data["data"].eq(gap)]
        data = data.loc[~(data["categoria"].eq("A") & (
            data["data"].lt(debut) | data["data"].eq(segment_gap)))]
        result = detect_segment_anomalies(data, configuration={"missing_policy": "zero"})
        a = result["points"].loc[result["points"]["segmento"].eq("A")].set_index("data")
        self.assertTrue(pd.isna(a.loc[gap, "valor"]))
        self.assertTrue(pd.isna(a.loc[debut - pd.Timedelta(days=1), "valor"]))
        self.assertEqual(a.loc[segment_gap, "valor"], 0)
        self.assertFalse(a.loc[segment_gap, "observado"])
        self.assertTrue(a.loc[segment_gap, "eligible"])

    def test_weekly_calendar_and_partial_edges_are_common(self):
        result = detect_segment_anomalies(
            sales(days=210, start="2024-01-03"), configuration={"frequency": "W-SUN"},
            min_train_observations=8)
        a = result["points"].loc[result["points"]["segmento"].eq("A")]
        b = result["points"].loc[result["points"]["segmento"].eq("B")]
        self.assertEqual(a["periodo"].tolist(), b["periodo"].tolist())
        self.assertEqual(a["split"].tolist(), b["split"].tolist())
        self.assertFalse(a.iloc[0]["eligible"])
        self.assertFalse(a.iloc[-1]["eligible"])

    def test_insufficient_calibration_is_explicit_and_not_a_normal_classification(self):
        data = sales()
        # A has full training support but no calibration observations.
        data = data.loc[~(data["categoria"].eq("A")
                          & data["data"].between(pd.Timestamp("2024-03-25"), pd.Timestamp("2024-05-05")))]
        result = detect_segment_anomalies(data)
        a = result["summary"].set_index("segmento").loc["A"]
        self.assertEqual(a["status"], "skipped")
        self.assertEqual(a["reason"], "insufficient_calibration_support")
        self.assertNotIn("A", set(result["points"]["segmento"]))

    def test_invalid_dimension_or_share_rejected(self):
        with self.assertRaises(ValueError):
            detect_segment_anomalies(sales(), dimension="produto")
        with self.assertRaises(ValueError):
            detect_segment_anomalies(sales(), min_train_share=1.1)

