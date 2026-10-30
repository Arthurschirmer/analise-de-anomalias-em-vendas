from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pandas as pd

from streamlit.testing.v1 import AppTest
from src.llm.mock_provider import MockLLMProvider


class ResearchViewsTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.provider = patch("src.ui.workspace.create_llm_provider_from_env", return_value=MockLLMProvider())
        self.provider.start()
        self.addCleanup(self.provider.stop)
        self.root = patch("src.ui.research_views.ROOT", Path(self.folder.name))
        self.root.start()
        self.addCleanup(self.root.stop)
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("No API calls in UI tests"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()

    def navigate(self, app, label):
        app.session_state["workspace_view"] = label
        return app.run()

    def submit(self, app, form):
        button = next(b for b in app.button if (b.key or "").startswith(f"FormSubmitter:{form}-"))
        return button.click().run()

    def test_short_history_demo_calculation_and_review_persistence(self):
        app = self.app()
        self.navigate(app, "Histórico")
        self.assertTrue(any("base é curta" in item.value for item in app.info))
        app.button(key="load_historical_demo").click().run()
        data = app.session_state["data"].copy()
        mask = data["data"].eq(data["data"].max())
        data.loc[mask, "faturamento"] *= 4
        data.loc[mask, "preco_unitario"] *= 4
        app.session_state["data"] = data
        app.run()
        self.submit(app, "historical_detection")
        self.assertEqual([e.message for e in app.exception], [])
        frame, calibration, config = app.session_state["historical_result"]
        self.assertTrue(calibration["frozen"])
        self.assertTrue(frame.iloc[-1]["flag"])
        next(w for w in app.text_input if w.label == "Responsável pela revisão").set_value("Arthur")
        self.submit(app, "human_alert_review")
        self.assertTrue(any("Revisão salva" in s.value for s in app.success))
        self.assertEqual(len(list(Path(self.folder.name, "outputs/reviews").glob("*.json"))), 1)
        self.navigate(app, "Painel")
        app.selectbox(key="demo_scenario").set_value("Mudança nos preços").run()
        app.button(key="load_demo").click().run()
        with self.assertRaises(KeyError):
            app.session_state["historical_result"]

    def test_benchmark_has_labeled_metrics_and_downloads(self):
        app = self.app()
        self.navigate(app, "Avaliação")
        self.submit(app, "benchmark_settings")
        self.assertEqual([e.message for e in app.exception], [])
        result = app.session_state["benchmark_result"]
        self.assertTrue(result["stages"]["controlled"]["ground_truth_complete"])
        self.assertTrue(Path(result["outputs"]["summary.json"]).is_file())

    def test_offline_comparison_pending_then_human_review(self):
        app = self.app()
        app.button(key="investigate_focus").click().run()
        self.submit(app, "manual_investigation")
        self.navigate(app, "Avaliação")
        app.selectbox(key="evaluation_mode").set_value("Relatório local e IA").run()
        app.button(key="compare_explanations").click().run()
        self.assertEqual([e.message for e in app.exception], [])
        result = app.session_state["explanation_comparison"]
        self.assertEqual(result["ai"]["status"], "not_run")
        self.assertEqual(result["human_review"]["status"], "pending")
        for name in ["fidelity", "clarity", "usefulness"]:
            app.selectbox(key="grade_local_" + name).set_value(4)
        app.text_input(key="explanation_reviewer").set_value("Arthur")
        self.submit(app, "human_explanation_review")
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(app.session_state["explanation_comparison"]["human_review"]["status"], "completed")

    def test_segment_category_and_region_views_use_train_selection(self):
        import numpy as np
        app = self.app()
        self.navigate(app, "Histórico")
        app.button(key="load_historical_demo").click().run()
        data = app.session_state["data"].copy()
        for col in ["faturamento", "preco_unitario"]:
            data[col] = data[col].astype(float)
        before = data.groupby("data")["faturamento"].sum()
        last = data["data"].max()
        for date in sorted(data.loc[data["data"].ge(last - pd.Timedelta(days=3)), "data"].unique()):
            target = data["data"].eq(date) & data["categoria"].eq("Casa")
            other = data["data"].eq(date) & ~data["categoria"].eq("Casa")
            lost = float(data.loc[target, "faturamento"].sum()) * .7
            data.loc[target, "faturamento"] *= .3
            data.loc[other, "faturamento"] *= 1 + lost / data.loc[other, "faturamento"].sum()
        data["preco_unitario"] = data["faturamento"] / data["quantidade"]
        self.assertTrue(np.allclose(before, data.groupby("data")["faturamento"].sum()))
        app.session_state["data"] = data
        app.run()
        app.selectbox(key="historical_scope").set_value("Por categoria").run()
        self.submit(app, "historical_detection")
        self.assertEqual([e.message for e in app.exception], [])
        result, config = app.session_state["segment_result"]
        self.assertEqual(result["metadata"]["selection_policy"], "training_support_only")
        casa = result["points"].loc[result["points"]["segmento"].eq("Casa")]
        self.assertTrue(casa.loc[casa["split"].eq("test"), "flag"].any())
        self.assertEqual(config["calibration_method"], "robust")
        app.selectbox(key="historical_scope").set_value("Por região / UF").run()
        self.submit(app, "historical_detection")
        self.assertEqual([e.message for e in app.exception], [])
        result, _ = app.session_state["segment_result"]
        self.assertEqual(set(result["summary"]["segmento"]), {"SP", "PR"})

    def test_sample_without_alerts_can_be_judged_and_completed(self):
        from src.evaluation.manual_review import read_reviews, summarize_review_sample
        app = self.app()
        self.navigate(app, "Histórico")
        app.button(key="load_historical_demo").click().run()
        self.submit(app, "historical_detection")
        app.number_input(key="review_sample_size").set_value(2)
        self.submit(app, "review_sample_settings")
        self.assertEqual([e.message for e in app.exception], [])
        panel = app.session_state["review_panel"]
        self.assertEqual(len(panel["sample"]), 2)
        self.assertFalse(panel["sample"]["flag"].any())
        self.navigate(app, "Avaliação")
        app.selectbox(key="evaluation_mode").set_value("Revisão humana").run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertFalse(any("Decisão do detector:" in item.value for item in app.caption))
        for index, review_id in enumerate(panel["sample"]["review_id"]):
            app.selectbox(key="review_case_id").set_value(review_id).run()
            next(w for w in app.selectbox if w.label == "Parecer sobre o ponto").set_value("Comportamento esperado")
            app.text_input(key="sample_review_reviewer").set_value("Revisor de teste")
            next(w for w in app.text_area if w.label == "Justificativa e evidências do parecer").set_value("Parecer fictício exclusivo do teste automatizado.")
            self.submit(app, "sample_case_review")
            self.assertEqual([e.message for e in app.exception], [])
            reviews = read_reviews(panel["scope"], panel["output_dir"])
            summary = summarize_review_sample(panel["sample"], reviews)
            if index == 0:
                self.assertIsNone(summary["metrics"])
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["metrics"]["is_estimate"])
        self.assertEqual(summary["metrics"]["accuracy"], 1.)

    def test_suite_ui_renders_metrics_and_chart_from_actual_small_run(self):
        from src.evaluation.benchmark_suite import run_benchmark_suite
        result = run_benchmark_suite(Path(self.folder.name, "suite"),
                                     seeds=(3, 11), days=180,
                                     scenario_names=["mild_drop", "gradual_drop", "category_hidden_drop"])
        app = self.app()
        app.session_state["benchmark_suite_result"] = result
        self.navigate(app, "Avaliação")
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(result["seeds"], [3, 11])
        self.assertTrue(any("Médias e desvios" in item.value for item in app.caption))

    def test_weekly_segments_accept_eight_observed_training_weeks(self):
        from src.evaluation.anomaly_benchmark import controlled_sales
        from src.ui.research_views import calculate_segments
        configuration = {"dimension": "categoria", "frequency": "W-SUN", "metric": "faturamento",
            "window": 8, "min_history": 4, "trend": "linear", "quantile": .98,
            "missing_policy": "unknown", "calibration_method": "robust",
            "max_segments": 12, "min_train_share": .01}
        result = calculate_segments(controlled_sales(days=210), configuration)
        self.assertEqual(result["metadata"]["evaluated_segments"], 2)
        self.assertEqual(result["configuration"]["min_train_observations"], 8)

    def test_review_discloses_assumed_zero_instead_of_observed_sales(self):
        from src.analysis.historical_detection import build_time_series, temporal_split, score_historical_series
        from src.evaluation.anomaly_benchmark import controlled_sales
        from src.evaluation.manual_review import create_review_sample
        data = controlled_sales(days=180)
        missing_date = data["data"].max()
        # Keep both calendar boundaries observed, but remove an interior date.
        missing_date -= pd.Timedelta(days=2)
        data = data.loc[data["data"].ne(missing_date)]
        series = temporal_split(build_time_series(data, missing_policy="zero"))
        scored = score_historical_series(series, trend="linear")
        sample = create_review_sample(scored, sample_size=100, blind=True)
        zero = sample.loc[~sample["observado"]].iloc[0]
        app = self.app()
        app.session_state["review_panel"] = {"scope": "c" * 64, "sample": sample,
            "configuration": {"missing_policy": "zero"}, "output_dir": str(Path(self.folder.name, "reviews"))}
        self.navigate(app, "Avaliação")
        app.selectbox(key="evaluation_mode").set_value("Revisão humana").run()
        app.selectbox(key="review_case_id").set_value(zero["review_id"]).run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertTrue(any("preenchimento de dias sem registros" in item.value for item in app.info))
        self.assertTrue(any(metric.label == "Valor analisado" for metric in app.metric))



