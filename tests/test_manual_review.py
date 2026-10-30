from tempfile import TemporaryDirectory
import unittest
from src.evaluation.anomaly_benchmark import controlled_sales
from src.evaluation.manual_review import read_reviews, review_scope, save_review


class ManualReviewTest(unittest.TestCase):
    def test_scoped_to_dataset_and_settings_and_persisted(self):
        data = controlled_sales(days=140)
        scope = review_scope(data, {"window": 8})
        self.assertNotEqual(scope, review_scope(data, {"window": 12}))
        self.assertEqual(scope, review_scope(data.sample(frac=1, random_state=2), {"window": 8}))
        with TemporaryDirectory() as folder:
            self.assertEqual(read_reviews(scope, folder), {})
            save_review(scope, "2024-01-01", "Inconclusivo", "Pesquisador", "Requer contexto", folder)
            self.assertEqual(read_reviews(scope, folder)["2024-01-01"]["judgement"], "Inconclusivo")
            self.assertEqual(read_reviews(review_scope(data, {}), folder), {})
            with self.assertRaises(ValueError):
                save_review(scope, "2024-01-02", "Anomalia confirmada", "", output_dir=folder)

