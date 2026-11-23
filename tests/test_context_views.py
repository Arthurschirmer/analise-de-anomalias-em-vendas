from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider


class ContextViewsTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for target, options in [
            ("src.ui.workspace.create_llm_provider_from_env", {"return_value": MockLLMProvider()}),
            ("src.ui.context_views.ROOT", {"new": Path(self.folder.name)}),
            ("urllib.request.urlopen", {"side_effect": AssertionError("No API calls in contextual UI")}),
        ]:
            item = patch(target, **options)
            item.start()
            self.addCleanup(item.stop)

    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
        app.session_state["workspace_view"] = "Avaliação"
        app.run()
        app.selectbox(key="evaluation_mode").set_value("Promoções e quedas").run()
        self.assertEqual([error.message for error in app.exception], [])
        return app

    def submit(self, app):
        button = next(item for item in app.button if (item.key or "").startswith("FormSubmitter:context_experiment_settings-"))
        return button.click().run()

    def test_generates_compares_and_inspects_days_without_network(self):
        app = self.app()
        app.text_input(key="context_experiment_seeds").set_value("42")
        self.submit(app)
        self.assertEqual([error.message for error in app.exception], [])
        result = app.session_state["context_experiment_result"]
        self.assertEqual(result["seeds"], [42])
        self.assertTrue(Path(result["outputs"]["calendario.csv"]).exists())
        self.assertEqual(len(app.dataframe[0].value), 2)
        self.assertTrue(any("Calendário informado" in item.value for item in app.markdown))
        app.segmented_control(key="context_result_method").set_value("Sem calendário").run()
        self.assertEqual([error.message for error in app.exception], [])
        app.selectbox(key="demo_scenario").set_value("Mudança nos preços").run()
        app.button(key="load_demo").click().run()
        with self.assertRaises(KeyError):
            app.session_state["context_experiment_result"]

    def test_invalid_seeds_do_not_run_an_experiment(self):
        app = self.app()
        app.text_input(key="context_experiment_seeds").set_value("42,42")
        self.submit(app)
        self.assertEqual([error.message for error in app.exception], [])
        self.assertTrue(any("distintas" in item.value for item in app.warning))
        with self.assertRaises(KeyError):
            app.session_state["context_experiment_result"]
        self.assertFalse(Path(self.folder.name, "outputs/contextual").exists())

    def test_demo_loads_sales_calendar_and_matching_evaluation(self):
        app = self.app()
        app.selectbox(key="demo_scenario").set_value("Promoções e quedas").run()
        app.button(key="load_demo").click().run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual(app.session_state["data_label"], "Promoções e quedas")
        self.assertEqual(len(app.session_state["data"]), 1680)
        self.assertEqual(len(app.session_state["context_calendar"]), 420)
        self.assertEqual(app.session_state["context_experiment_result"]["mode"], "offline_synthetic_fixture")
        self.assertEqual(app.session_state["workspace_view"], "Histórico")
        self.assertEqual(app.segmented_control(key="primary_menu").value, "Analisar")
        self.assertFalse(any(item.label == "Variações sinalizadas" for item in app.metric))
        app.segmented_control(key="workflow_mode").set_value("Experimentos do TCC").run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertTrue(any("arquivos da base fictícia carregada" in item.value for item in app.caption))
        self.assertTrue(any(item.key == "recalculate_context_fixture" for item in app.button))
        app.button(key="recalculate_context_fixture").click().run()
        self.assertEqual([error.message for error in app.exception], [])
        app.selectbox(key="demo_scenario").set_value("Queda nas vendas").run()
        app.button(key="load_demo").click().run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertIsNone(app.session_state["context_calendar"])
        self.assertIsNone(app.session_state["context_fixture_dir"])

    def test_direct_link_opens_context_demo_on_a_fresh_session(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
        app.query_params["demo_scenario"] = "Promoções e quedas"
        app.run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual(app.session_state["data_label"], "Promoções e quedas")
        self.assertEqual(len(app.session_state["data"]), 1680)
        self.assertEqual(app.session_state["workspace_view"], "Histórico")
        self.assertEqual(app.segmented_control(key="workflow_mode").value, "Analisar dados")
        self.assertEqual(app.segmented_control(key="primary_menu").value, "Analisar")
        self.assertEqual(app.session_state["context_experiment_result"]["seeds"], [42])

    def test_previous_context_session_moves_to_plain_analysis_without_losing_results(self):
        app = self.app()
        app.selectbox(key="demo_scenario").set_value("Promoções e quedas").run()
        app.button(key="load_demo").click().run()
        before = app.session_state["context_experiment_result"]
        app.session_state["workspace_view"] = "Avaliação"
        del app.session_state["simple_flow_initialized"]
        app.run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual(app.session_state["workspace_view"], "Histórico")
        self.assertEqual(app.segmented_control(key="workflow_mode").value, "Analisar dados")
        self.assertEqual(app.session_state["context_experiment_result"], before)

    def test_old_demo_schema_reloads_sales_calendar_and_results(self):
        app = self.app()
        app.selectbox(key="demo_scenario").set_value("Promoções e quedas").run()
        app.button(key="load_demo").click().run()
        expected_total = app.session_state["data"]["faturamento"].sum()
        # Simula um estado salvo antes de renderizar os widgets da nova versão.
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
        app.session_state["context_fixture_dir"] = str(Path(__file__).resolve().parents[1] / "data/contextual_demo")
        app.session_state["startup_attempted"] = True
        app.session_state["context_experiment_result"] = {"version": 1}
        app.session_state["data_label"] = "Demonstração anterior"
        app.session_state["demo_scenario"] = "Demonstração anterior"
        app.session_state["evaluation_mode"] = "Experimento anterior"
        app.session_state["alert_chat_context"] = {"period": "anterior"}
        app.run()
        self.assertEqual([error.message for error in app.exception], [])
        self.assertEqual(app.session_state["data_label"], "Promoções e quedas")
        self.assertEqual(app.selectbox(key="demo_scenario").value, "Promoções e quedas")
        self.assertEqual(list(app.session_state["context_calendar"].columns), ["data", "promocao"])
        self.assertEqual(app.session_state["context_experiment_result"]["version"], 2)
        self.assertEqual(app.session_state["data"]["faturamento"].sum(), expected_total)
        self.assertIsNone(app.session_state["alert_chat_context"])
