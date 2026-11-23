from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.evaluation.contextual_benchmark import run_contextual_fixture


SCRIPT = """
from src.ui.context_views import render_loaded_context_analysis
render_loaded_context_analysis()
"""
ROOT = Path(__file__).resolve().parents[1]


class ContextAnalysisViewTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.fixture = ROOT / "data/contextual_demo"
        self.result = run_contextual_fixture(self.fixture, Path(self.folder.name) / "results")
        self.path = Path(self.result["outputs"]["point_results.csv"])
        self.points = pd.read_csv(self.path, parse_dates=["data"])
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("No network in analysis UI"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def app(self):
        app = AppTest.from_string(SCRIPT, default_timeout=20)
        app.session_state["context_fixture_dir"] = str(self.fixture)
        app.session_state["context_experiment_result"] = deepcopy(self.result)
        app.run()
        self.assertEqual([error.message for error in app.exception], [])
        return app

    def test_summary_matches_predictions_and_hides_experiment_fields(self):
        expected = self.points.loc[self.points["detector"].eq("contextual") & self.points["split"].eq("test")]
        eligible = expected["eligible"].astype(bool)
        flagged = eligible & expected["flag"].astype(bool)
        app = self.app()
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Dias analisados"], str(int(eligible.sum())))
        self.assertEqual(metrics["Dias para investigar"], str(int(flagged.sum())))
        self.assertEqual(metrics["Dias sem decisão"], str(int((~eligible).sum())))
        self.assertEqual(len(app.dataframe[0].value), int(flagged.sum()))
        self.assertEqual(app.selectbox(key="context_selected_day").value, expected.loc[flagged, "data"].iloc[0])
        visible = "\n".join(item.value for items in (app.markdown, app.caption) for item in items).lower()
        self.assertNotIn("gabarito", visible)
        self.assertNotIn("precisão", visible)
        self.assertNotIn("f1", visible)
        self.assertNotIn("semente", visible)
        self.assertFalse(any("gabarito" in item.label.lower() for item in app.expander))
        self.assertTrue(any("Referência para este dia" in item.value for item in app.markdown))

    def test_answer_key_columns_are_not_needed_for_the_main_view(self):
        self.points.drop(columns=["is_anomaly", "event_id"], errors="ignore").to_csv(self.path, index=False)
        app = self.app()
        self.assertEqual(app.metric[1].value, "5")
        self.assertEqual([error.message for error in app.exception], [])

    def test_day_without_decision_is_not_counted_or_presented_as_normal(self):
        mask = self.points["detector"].eq("contextual") & self.points["split"].eq("test") & self.points["flag"]
        row = self.points.loc[mask].index[0]
        date = self.points.loc[row, "data"]
        # Even an inconsistent external CSV flag must not override eligibility.
        self.points.loc[row, "eligible"] = False
        self.points.to_csv(self.path, index=False)
        app = self.app()
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Dias analisados"], "104")
        self.assertEqual(metrics["Dias para investigar"], "4")
        self.assertEqual(metrics["Dias sem decisão"], "1")
        app.selectbox(key="context_selected_day").set_value(date).run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertTrue(any("sem classificação" in item.value for item in app.info))
        self.assertFalse(app.success)

    def test_reanalysis_uses_the_loaded_fixture_and_refreshes_result(self):
        changed = self.points.copy()
        changed.loc[changed["detector"].eq("contextual"), "flag"] = False
        changed_path = Path(self.folder.name) / "changed_points.csv"
        changed.to_csv(changed_path, index=False)
        changed_result = deepcopy(self.result)
        changed_result["outputs"]["point_results.csv"] = str(changed_path)
        app = self.app()
        with patch("src.ui.context_views.evaluate_loaded_fixture", return_value=changed_result) as evaluate:
            app.button(key="reanalyze_loaded_context").click().run()
        evaluate.assert_called_once_with(str(self.fixture))
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual(app.metric[1].value, "0")
        self.assertTrue(any("Nenhum dos dias avaliados" in item.value for item in app.info))
