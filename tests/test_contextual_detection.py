import unittest

import numpy as np
import pandas as pd

from src.analysis.contextual_detection import _robust_fit, analyze_contextual_series


def example():
    dates = pd.date_range("2023-01-02", periods=420, freq="D")
    positions = np.arange(len(dates))
    promotion = positions % 17 < 3
    weekday = np.array([1.0, 1.04, 0.99, 1.08, 1.12, 0.82, 0.68])[dates.dayofweek]
    values = 1000 * np.exp(0.001 * positions) * weekday * 1.35 ** promotion
    series = pd.DataFrame({"data": dates, "valor": values, "completo": True,
                           "split": ["train"] * 210 + ["calibration"] * 105 + ["test"] * 105})
    calendar = pd.DataFrame({"data": dates, "promocao": promotion})
    return series, calendar


class ContextualDetectionTest(unittest.TestCase):
    def test_huber_location_satisfies_the_estimating_equation(self):
        # For this intercept-only example MAD(y - location) = 2. The first
        # six residuals stay in Huber's quadratic region; the last is clipped.
        # Thus sum(psi) = 0 gives 6 * location = sum(y[:6]) + 1.345 * sigma.
        values = np.array([-2., -1., 0., 1., 2., 3., 50.])
        coefficients, _ = _robust_fit(np.ones((len(values), 1)), values)
        sigma = 1.4826 * 2.0
        expected_location = (values[:6].sum() + 1.345 * sigma) / 6.0
        self.assertAlmostEqual(coefficients[0], expected_location, places=8)
        standardized = (values - coefficients[0]) / sigma
        self.assertAlmostEqual(float(np.clip(standardized, -1.345, 1.345).sum()), 0.0, places=8)

    def test_accounts_for_known_events_but_finds_an_unexpected_drop(self):
        series, calendar = example()
        series.loc[350, "valor"] *= 0.45
        result = analyze_contextual_series(series, calendar)
        self.assertTrue(result["available"])
        scored = result["scored"]
        alerts = scored.loc[scored["split"].eq("test") & scored["flag"], "data"]
        self.assertEqual(alerts.tolist(), [series.loc[350, "data"]])
        self.assertFalse(scored.loc[scored["split"].eq("train"), "eligible"].any())
        self.assertAlmostEqual(result["model"]["context_factors"]["promocao"], 1.35, places=7)
        self.assertEqual(set(result["model"]["context_factors"]), {"promocao"})

    def test_test_values_and_calendar_cannot_refit_model_or_threshold(self):
        series, calendar = example()
        original = analyze_contextual_series(series, calendar)
        changed_series, changed_calendar = series.copy(), calendar.copy()
        changed_series.loc[315:, "valor"] *= 0.2
        changed_calendar.loc[315:, "promocao"] = ~changed_calendar.loc[315:, "promocao"]
        modified = analyze_contextual_series(changed_series, changed_calendar)
        self.assertEqual(original["model"], modified["model"])
        self.assertEqual(original["calibration"], modified["calibration"])
        np.testing.assert_allclose(original["scored"].loc[:314, "expected"],
                                   modified["scored"].loc[:314, "expected"])

    def test_labels_and_extra_numeric_columns_are_not_used_as_features(self):
        series, calendar = example()
        original = analyze_contextual_series(series, calendar)
        calendar["anomalia_injetada"] = np.arange(420) % 2 == 0
        calendar["efeito_conhecido"] = np.arange(420) * 10000
        modified = analyze_contextual_series(series, calendar)
        self.assertEqual(original["model"], modified["model"])
        self.assertEqual(original["calibration"], modified["calibration"])
        self.assertNotIn("anomalia_injetada", modified["scored"])

    def test_missing_calendar_preserves_dates_and_withholds_decisions(self):
        series, calendar = example()
        result = analyze_contextual_series(series, calendar.iloc[:315])
        self.assertTrue(result["available"])
        scored = result["scored"]
        self.assertEqual(len(scored), len(series))
        self.assertEqual(scored["data"].tolist(), series["data"].tolist())
        test = scored.loc[scored["split"].eq("test")]
        self.assertFalse(test["eligible"].any())
        self.assertFalse(test["flag"].any())
        self.assertTrue(test["reason"].eq("missing_context").all())
        self.assertTrue(test["promocao"].isna().all())

    def test_nullable_event_is_not_interpreted_as_false(self):
        series, calendar = example()
        calendar["promocao"] = calendar["promocao"].astype("boolean")
        calendar.loc[350, "promocao"] = pd.NA
        scored = analyze_contextual_series(series, calendar)["scored"]
        self.assertEqual(scored.loc[350, "reason"], "missing_context")
        self.assertFalse(scored.loc[350, "eligible"])

    def test_context_confounded_with_weekday_is_declared_unavailable(self):
        series, calendar = example()
        calendar["promocao"] = calendar["data"].dt.dayofweek.eq(1)
        result = analyze_contextual_series(series, calendar)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "rank_deficient_context")
        self.assertFalse(result["scored"]["flag"].any())

    def test_context_never_seen_in_training_cannot_have_an_invented_effect(self):
        series, calendar = example()
        calendar.loc[:209, "promocao"] = False
        result = analyze_contextual_series(series, calendar)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "insufficient_context_history:promocao")
        self.assertFalse(result["scored"]["eligible"].any())

    def test_zero_is_evaluated_but_negative_and_missing_values_are_not(self):
        series, calendar = example()
        series.loc[350, "valor"] = 0.0
        series.loc[351, "valor"] = -10.0
        series.loc[352, "valor"] = np.nan
        scored = analyze_contextual_series(series, calendar)["scored"]
        self.assertEqual(scored.loc[350, "reason"], "ready")
        self.assertTrue(scored.loc[350, "eligible"])
        self.assertTrue(scored.loc[350, "flag"])
        self.assertGreater(scored.loc[350, "expected"], 0)
        self.assertEqual(scored.loc[351, "reason"], "negative_observation")
        self.assertEqual(scored.loc[352, "reason"], "missing_observation")
        self.assertFalse(scored.loc[[351, 352], "eligible"].any())

    def test_zero_training_value_is_excluded_from_the_log_fit(self):
        series, calendar = example()
        series.loc[20, "valor"] = 0.0
        result = analyze_contextual_series(series, calendar)
        self.assertTrue(result["available"])
        self.assertEqual(result["model"]["n_train"], 209)
        self.assertAlmostEqual(result["model"]["context_factors"]["promocao"], 1.35, places=7)
        self.assertFalse(result["scored"].loc[20, "eligible"])

    def test_invalid_context_and_duplicate_dates_are_rejected(self):
        series, calendar = example()
        invalid = calendar.astype({"promocao": "object"})
        invalid.loc[0, "promocao"] = "False"
        with self.assertRaises(ValueError):
            analyze_contextual_series(series, invalid)
        with self.assertRaises(ValueError):
            analyze_contextual_series(series, pd.concat([calendar, calendar.iloc[:1]]))


if __name__ == "__main__":
    unittest.main()
