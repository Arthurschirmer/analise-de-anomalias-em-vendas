from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider


class WorkspaceFlowTest(unittest.TestCase):
    def setUp(self):
        provider = patch("src.ui.workspace.create_llm_provider_from_env", return_value=MockLLMProvider())
        provider.start()
        self.addCleanup(provider.stop)
        network = patch("urllib.request.urlopen", side_effect=AssertionError("No API calls in workflow tests"))
        network.start()
        self.addCleanup(network.stop)

    def app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()

    def check(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def test_start_analysis_selects_months_for_short_demo_and_keeps_monthly_cards_on_overview(self):
        app = self.app()
        self.check(app)
        self.assertEqual(app.segmented_control(key="primary_menu").options, list(["Visão geral", "Analisar", "Conversar", "Dados e resultados"]))
        self.assertTrue(any(item.label == "Variações sinalizadas" for item in app.metric))
        app.button(key="start_analysis").click().run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Investigação")
        self.assertEqual(app.segmented_control(key="primary_menu").value, "Analisar")
        self.assertFalse(any(item.label == "Variações sinalizadas" for item in app.metric))

    def test_primary_and_academic_navigation_preserve_loaded_data(self):
        app = self.app()
        original = app.session_state["data"].copy()
        app.segmented_control(key="primary_menu").set_value("Conversar").run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Assistente")
        self.assertFalse(any(item.label == "Faturamento da base" for item in app.metric))
        app.segmented_control(key="workflow_mode").set_value("Experimentos do TCC").run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Avaliação")
        self.assertEqual(app.session_state["data"].to_csv(index=False), original.to_csv(index=False))
        app.segmented_control(key="workflow_mode").set_value("Analisar dados").run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Painel")
        app.segmented_control(key="primary_menu").set_value("Dados e resultados").run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Dados e relatório")

    def test_contextual_alert_opens_chat_with_calendar_without_monthly_routing(self):
        app = self.app()
        app.selectbox(key="demo_scenario").set_value("Promoções e quedas")
        app.button(key="load_demo").click().run()
        self.check(app)
        self.assertTrue(any(item.label == "Baixar alerta em PDF" for item in app.get("download_button")))
        app.button(key="context_chat_about_alert").click().run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Assistente")
        self.assertTrue(app.session_state["alert_chat_context"].get("calendar_context"))
        with patch("src.ui.workspace._provider_for_request", side_effect=AssertionError("No API for alert conversation")):
            app.chat_input(key="chat_prompt").set_value("A promoção explica este alerta?").run()
        self.check(app)
        self.assertIn("referência já considera", app.session_state["messages"][-1]["content"])
