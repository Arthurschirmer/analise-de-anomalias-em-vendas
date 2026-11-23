import unittest

import numpy as np
import pandas as pd

from src.data.contextual_demo import generate_contextual_sales
from src.data.loader import STANDARD_COLUMNS, prepare_sales_data


class ContextualDemoTest(unittest.TestCase):
    def test_reproducibility_and_independent_predictors(self):
        first = generate_contextual_sales(seed=42)
        repeated = generate_contextual_sales(seed=42)
        other = generate_contextual_sales(seed=137)
        for name in ("sales", "calendar", "truth"):
            pd.testing.assert_frame_equal(getattr(first, name), getattr(repeated, name))
        self.assertEqual(first.metadata, repeated.metadata)
        self.assertFalse(first.sales.equals(other.sales))
        pd.testing.assert_frame_equal(first.calendar, other.calendar)
        pd.testing.assert_series_equal(first.truth.is_anomaly, other.truth.is_anomaly)
        self.assertEqual(list(first.calendar.columns), ["data", "promocao"])
        self.assertEqual(first.sales.attrs, {})
        self.assertEqual(first.calendar.attrs, {})
        self.assertEqual(list(first.sales.columns), STANDARD_COLUMNS)
        self.assertNotIn("is_anomaly", first.sales.columns)

    def test_sales_reconcile_measured_discount_and_integer_quantities(self):
        dataset = generate_contextual_sales()
        data = prepare_sales_data(dataset.sales)
        self.assertEqual(len(data), 420 * 4)
        self.assertTrue((data.quantidade > 0).all())
        self.assertTrue((data.quantidade % 1 == 0).all())
        self.assertTrue(np.allclose(
            data.faturamento, data.quantidade * data.preco_unitario * (1 - data.desconto),
            rtol=0, atol=1e-8,
        ))
        promo_by_day = dataset.calendar.set_index("data").promocao
        is_promo = data.data.map(promo_by_day)
        self.assertTrue(data.loc[is_promo, "desconto"].eq(.10).all())
        self.assertTrue(data.loc[~is_promo, "desconto"].eq(0).all())
        daily = data.groupby("data")[["quantidade", "faturamento"]].sum()
        truth = dataset.truth.set_index("data")
        self.assertTrue(np.array_equal(daily.quantidade, truth.quantity))
        self.assertTrue(np.allclose(daily.faturamento, truth.revenue))

    def test_all_contexts_are_repeated_in_every_split_without_training_anomalies(self):
        for days in (280, 420):
            dataset = generate_contextual_sales(days=days)
            expected_contexts = {"normal", "promotion"}
            for split in ("train", "calibration", "test"):
                part = dataset.truth.loc[dataset.truth.split.eq(split)]
                self.assertEqual(set(part.expected_context_event), expected_contexts)
            self.assertFalse(dataset.truth.loc[dataset.truth.split.ne("test"), "is_anomaly"].any())
            anomalies = dataset.truth.loc[dataset.truth.is_anomaly]
            self.assertEqual(len(anomalies), 5)
            self.assertEqual(set(anomalies.expected_context_event), expected_contexts)
            self.assertEqual(anomalies.expected_context_event.value_counts().to_dict(), {"promotion": 3, "normal": 2})
            self.assertTrue(anomalies.event_id.str.startswith("unexpected_drop_").all())
            expected_days = dataset.truth.loc[~dataset.truth.expected_context_event.eq("normal") & ~dataset.truth.is_anomaly]
            self.assertGreater(len(expected_days.loc[expected_days.split.eq("test")]), 0)
        self.assertEqual(dataset.metadata["splits"]["train"]["count"], 210)
        self.assertEqual(dataset.metadata["splits"]["calibration"]["count"], 105)
        self.assertEqual(dataset.metadata["splits"]["test"]["count"], 105)

    def test_nominal_effects_and_realized_rounding_are_recorded_separately(self):
        dataset = generate_contextual_sales()
        truth = dataset.truth
        promotion = truth.loc[truth.expected_context_event.eq("promotion")]
        normal = truth.loc[truth.expected_context_event.eq("normal")]
        self.assertTrue(np.allclose(promotion.nominal_context_quantity_change_pct, 45))
        self.assertTrue(np.allclose(promotion.nominal_context_revenue_change_pct, 30.5))
        self.assertTrue(np.allclose(normal.nominal_context_quantity_change_pct, 0))
        self.assertTrue(np.allclose(normal.nominal_context_revenue_change_pct, 0))
        target = truth.baseline_quantity * (1 + truth.nominal_context_quantity_change_pct / 100)
        self.assertTrue(np.all(abs(truth.context_quantity - target) <= .50000001))
        surprise = truth.loc[truth.is_anomaly]
        self.assertTrue(np.allclose(surprise.nominal_unexpected_quantity_change_pct, -50))
        self.assertTrue(np.all(abs(surprise.quantity - surprise.context_quantity * .5) <= .5))
        self.assertTrue((surprise.revenue < surprise.context_revenue).all())
        ordinary = truth.loc[~truth.is_anomaly]
        self.assertTrue(ordinary.quantity.eq(ordinary.context_quantity).all())
        self.assertTrue(np.allclose(ordinary.revenue, ordinary.context_revenue))
        self.assertIn("assumptions", dataset.metadata["causality_note"])

    def test_rejects_insufficient_calendar_and_invalid_seed_or_date(self):
        for options in ({"days": 279}, {"days": True}, {"seed": True},
                        {"seed": -1}, {"seed": 2**32}, {"start": "not-a-date"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate_contextual_sales(**options)


if __name__ == "__main__":
    unittest.main()
