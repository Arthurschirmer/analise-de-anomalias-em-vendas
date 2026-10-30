import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.evaluation.manual_review import (
    create_review_sample, read_review_sample, read_reviews, save_review,
    save_review_sample, summarize_review_sample,
)


class ReviewSamplingTest(unittest.TestCase):
    def frame(self, flags=10, unflagged=90):
        size = flags + unflagged
        dates = pd.date_range("2024-01-01", periods=size)
        return pd.DataFrame({
            "data": dates, "periodo": dates.strftime("%Y-%m-%d"), "split": "test",
            "eligible": True, "flag": [True] * flags + [False] * unflagged,
            "valor": np.arange(size, dtype=float) + 100,
            "expected": 100., "lower": 80., "upper": 120.,
        })

    def reviews(self, sample, inconclusive=False):
        result = {}
        for stratum, part in sample.groupby("stratum"):
            for i, row in enumerate(part.to_dict(orient="records")):
                judgement = "Anomalia confirmada" if i == 0 else "Comportamento esperado"
                result[row["review_id"]] = {"judgement": judgement, "reviewer": "Pesquisador"}
        if inconclusive and len(sample):
            result[sample.iloc[0]["review_id"]]["judgement"] = "Inconclusivo"
        return result

    def test_strata_are_sampled_without_duplicates_with_design_weights(self):
        sample = create_review_sample(self.frame(flags=3, unflagged=97), sample_size=10, seed=7)
        self.assertEqual(len(sample), 10)
        self.assertEqual(sample["stratum"].value_counts().to_dict(), {"unflagged": 7, "flagged": 3})
        self.assertFalse(sample["review_id"].duplicated().any())
        for stratum, population in [("flagged", 3), ("unflagged", 97)]:
            part = sample.loc[sample["stratum"].eq(stratum)]
            self.assertTrue(np.allclose(part["sample_weight"], population / len(part)))
            self.assertTrue(np.allclose(part["inclusion_probability"], len(part) / population))
        self.assertNotIn("judgement", sample.columns)

    def test_reproducible_when_input_rows_are_permuted_and_seed_changes_panel(self):
        data = self.frame()
        first = create_review_sample(data, 10, seed=22)
        same = create_review_sample(data.sample(frac=1, random_state=4), 10, seed=22)
        pd.testing.assert_frame_equal(first, same)
        self.assertEqual(first.attrs, same.attrs)
        other = create_review_sample(data, 10, seed=23)
        self.assertNotEqual(first.attrs["sample_id"], other.attrs["sample_id"])
        self.assertNotEqual(first["review_id"].tolist(), other["review_id"].tolist())

    def test_training_calibration_and_ineligible_points_never_enter_panel(self):
        data = self.frame()
        data.loc[:20, "split"] = "train"
        data.loc[21:40, "split"] = "calibration"
        data.loc[41:50, "eligible"] = False
        sample = create_review_sample(data, 100)
        self.assertEqual(len(sample), 49)
        self.assertTrue(sample["split"].eq("test").all())
        self.assertTrue(sample["eligible"].all())
        self.assertTrue(set(sample["periodo"]).isdisjoint(set(data.loc[:50, "periodo"])))

    def test_panel_exists_without_any_alerts_and_can_reveal_false_negative(self):
        sample = create_review_sample(self.frame(flags=0, unflagged=20), 5)
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample.attrs["population_counts"], {"flagged": 0, "unflagged": 20})
        self.assertFalse(sample["flag"].any())
        review = self.reviews(sample)
        summary = summarize_review_sample(sample, review)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["observed_reviewed_confusion"]["fn"], 1)
        self.assertEqual(summary["metrics"]["recall"], 0)
        self.assertIsNone(summary["metrics"]["precision"])

    def test_only_alert_stratum_is_a_valid_finite_population(self):
        sample = create_review_sample(self.frame(flags=10, unflagged=0), 5)
        summary = summarize_review_sample(sample, self.reviews(sample))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["population_size"], 10)
        self.assertFalse(summary["is_census"])

    def test_partial_and_inconclusive_panels_withhold_all_metrics(self):
        sample = create_review_sample(self.frame(), 10)
        review = self.reviews(sample)
        del review[sample.iloc[0]["review_id"]]
        summary = summarize_review_sample(sample, review)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["missing"], 1)
        self.assertIsNone(summary["metrics"])
        summary = summarize_review_sample(sample, self.reviews(sample, inconclusive=True))
        self.assertEqual(summary["status"], "inconclusive")
        self.assertEqual(summary["inconclusive"], 1)
        self.assertIsNone(summary["metrics"])

    def test_complete_panel_uses_weights_not_naive_sample_recall(self):
        sample = create_review_sample(self.frame(flags=10, unflagged=90), 4)
        result = summarize_review_sample(sample, self.reviews(sample))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["observed_reviewed_confusion"], {"tp": 1, "fp": 1, "fn": 1, "tn": 1})
        metrics = result["metrics"]
        self.assertTrue(metrics["is_estimate"])
        self.assertEqual(metrics["weighted_confusion"], {"tp": 5., "fp": 5., "fn": 45., "tn": 45.})
        self.assertEqual(metrics["precision"], .5)
        self.assertAlmostEqual(metrics["recall"], .1)
        self.assertAlmostEqual(metrics["f1"], 1 / 6)

    def test_census_metrics_are_labelled_as_finite_eligible_test_census(self):
        sample = create_review_sample(self.frame(flags=2, unflagged=2), 100)
        result = summarize_review_sample(sample, self.reviews(sample))
        self.assertTrue(result["is_census"])
        self.assertFalse(result["metrics"]["is_estimate"])
        self.assertEqual(result["metrics"]["basis"], "eligible_test_census")
        self.assertAlmostEqual(result["metrics"]["recall"], .5)

    def test_explicitly_conflicting_panel_id_does_not_reuse_human_labels(self):
        sample = create_review_sample(self.frame(flags=2, unflagged=2), 100)
        reviews = self.reviews(sample)
        review_id = sample.iloc[0]["review_id"]
        reviews[review_id]["metadata"] = {"sample_id": "different_panel"}
        result = summarize_review_sample(sample, reviews)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing"], 1)
        self.assertIsNone(result["metrics"])
        reviews[review_id]["metadata"]["sample_id"] = sample.attrs["sample_id"]
        result = summarize_review_sample(sample, reviews)
        self.assertEqual(result["status"], "complete")
        self.assertIsNotNone(result["metrics"])

    def test_empty_population_and_invalid_requests(self):
        data = self.frame(flags=0, unflagged=0)
        sample = create_review_sample(data, 10)
        self.assertTrue(sample.empty)
        result = summarize_review_sample(sample, {})
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["metrics"])
        for size in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                create_review_sample(self.frame(), size)
        with self.assertRaises(ValueError):
            create_review_sample(self.frame(), 1)
        with self.assertRaises(ValueError):
            create_review_sample(self.frame(), seed=-1)
        duplicated = pd.concat([self.frame(), self.frame().iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            create_review_sample(duplicated)

    def test_segment_identity_and_manifest_and_judgement_persistence(self):
        data = self.frame(flags=2, unflagged=2)
        data["dimensao"] = "regiao"
        data["segmento"] = "SP"
        other = data.copy()
        other["segmento"] = "PR"
        sample = create_review_sample(pd.concat([data, other]), 8, seed=4, blind=True)
        self.assertEqual(sample["review_id"].nunique(), 8)
        self.assertTrue(sample.attrs["blind"])
        scope = "a" * 64
        with TemporaryDirectory() as directory:
            path = save_review_sample(scope, sample, directory)
            self.assertTrue(path.name.endswith("_sample.json"))
            loaded = read_review_sample(scope, directory)
            self.assertEqual(loaded.attrs, sample.attrs)
            self.assertEqual(loaded["review_id"].tolist(), sample["review_id"].tolist())
            pd.testing.assert_series_equal(loaded["data"], sample["data"], check_dtype=False)
            row = sample.iloc[0]
            save_review(scope, row["review_id"], "Inconclusivo", "Pesquisador", "Requer contexto",
                        directory, metadata={"periodo": row["periodo"], "sample_id": sample.attrs["sample_id"]})
            saved = read_reviews(scope, directory)[row["review_id"]]
            self.assertEqual(saved["periodo"], row["periodo"])
            self.assertEqual(saved["review_id"], row["review_id"])
            self.assertEqual(saved["metadata"]["sample_id"], sample.attrs["sample_id"])
            save_review(scope, row["review_id"], "Anomalia confirmada", "Revisor", "Fonte adicional",
                        directory, metadata={"periodo": row["periodo"]})
            latest = read_reviews(scope, directory)[row["review_id"]]
            self.assertEqual(latest["history"][0]["judgement"], "Inconclusivo")
            self.assertEqual(latest["judgement"], "Anomalia confirmada")

    def test_missing_manifest_returns_none_and_invalid_scope_rejected(self):
        with TemporaryDirectory() as directory:
            self.assertIsNone(read_review_sample("b" * 64, directory))
            with self.assertRaises(ValueError):
                read_review_sample("../outside", directory)

    def test_uncovered_stratum_prevents_estimates(self):
        sample = create_review_sample(self.frame(), 10)
        altered = sample.loc[sample["stratum"].eq("flagged")].copy()
        summary = summarize_review_sample(altered, self.reviews(altered))
        self.assertEqual(summary["status"], "missing_stratum")
        self.assertIsNone(summary["metrics"])


if __name__ == "__main__":
    unittest.main()

