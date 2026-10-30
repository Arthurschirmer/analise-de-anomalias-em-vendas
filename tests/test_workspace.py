"""Offline regressions for the complete anomaly investigation interface."""

from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
FAKE_SECRET = "gsk_OFFLINETESTONLY1234567890"


class BlockedRemoteProvider:
    def generate(self, system_prompt, user_prompt, context):
        raise RuntimeError(f"HTTP 403: error code: 1010; Bearer {FAKE_SECRET}")

    def interpret_query(self, question, context):
        raise RuntimeError(f"HTTP 403: error code: 1010; {FAKE_SECRET}")


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        # Every configured provider is replaced before the app starts. No real
        # .env credentials are read and no request is sent to an AI service.
        self.provider_patch = patch(
            "src.ui.workspace.create_llm_provider_from_env",
            return_value=MockLLMProvider(),
        )
        self.provider_factory = self.provider_patch.start()
        self.addCleanup(self.provider_patch.stop)

    def app(self):
        return AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    def assert_no_exceptions(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def navigate(self, app, view):
        app.session_state["workspace_view"] = view
        return app.run()

    def submit(self, app, form):
        button = next(
            item for item in app.button
            if (item.key or "").startswith(f"FormSubmitter:{form}-")
        )
        return button.click().run()

    def investigate_focus(self, app):
        app.button(key="investigate_focus").click().run()
        return self.submit(app, "manual_investigation")

    def test_initial_dashboard_loads_demo_and_calculated_indicators(self):
        app = self.app()
        self.assert_no_exceptions(app)
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Faturamento da base"], "R$ 125.921,00")
        self.assertEqual(metrics["Unidades vendidas"], "243")
        self.assertEqual(metrics["Variações sinalizadas"], "1")
        self.assertEqual(app.session_state["available_periods"], ["2024-07", "2024-08"])
        self.assertFalse(app.session_state["use_ai"])

    def test_focus_action_compares_actual_periods_and_result_survives_dimension_change(self):
        app = self.app()
        app.button(key="investigate_focus").click().run()
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["workspace_view"], "Investigação")
        self.assertEqual(app.selectbox(key="compare_previous").value, "2024-07")
        self.assertEqual(app.selectbox(key="compare_current").value, "2024-08")
        self.submit(app, "manual_investigation")
        self.assert_no_exceptions(app)
        report = app.session_state["last_investigation"].structured_report
        self.assertEqual(report["periodos"], {"periodo_anterior": "2024-07", "periodo_atual": "2024-08"})
        self.assertAlmostEqual(report["resumo_comparativo"]["variacao_faturamento"], -43097.6)
        self.assertTrue(report["causas_provaveis"])
        app.session_state["report_dimension"] = "Região"
        app.run()
        self.assert_no_exceptions(app)
        persisted = app.session_state["last_investigation"].structured_report
        self.assertEqual(persisted["resumo_comparativo"], report["resumo_comparativo"])
        self.assertTrue(any(item.value == "Resultado da investigação" for item in app.subheader))

    def test_no_anomaly_is_information_and_threshold_survives_navigation(self):
        app = self.app()
        self.navigate(app, "Investigação")
        app.number_input(key="threshold_input").set_value(1000.0)
        self.submit(app, "automatic_investigation")
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.error), 0)
        feedback = app.session_state["detection_feedback"]
        self.assertEqual(feedback["kind"], "info")
        self.assertEqual(feedback["code"], "no_anomaly")
        self.assertTrue(any("Nenhuma anomalia" in item.value for item in app.info))
        self.assertIsNone(app.session_state["last_investigation"])
        self.navigate(app, "Painel")
        self.assertEqual(app.session_state["threshold_pct"], 1000.0)
        self.assertEqual(next(item.value for item in app.metric if item.label == "Variações sinalizadas"), "0")
        self.navigate(app, "Investigação")
        self.assertEqual(app.number_input(key="threshold_input").value, 1000.0)

    def test_remote_1010_keeps_local_report_and_sanitizes_technical_details(self):
        self.provider_factory.return_value = BlockedRemoteProvider()
        app = self.app()
        self.assertTrue(app.session_state["use_ai"])
        self.navigate(app, "Investigação")
        self.submit(app, "automatic_investigation")
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(app.session_state["last_investigation_mode"], "local")
        feedback = app.session_state["last_investigation_feedback"]
        self.assertEqual(feedback["code"], "cloudflare_1010")
        report = app.session_state["last_investigation"].structured_report
        self.assertAlmostEqual(report["resumo_comparativo"]["faturamento_atual"], 41411.7)
        self.assertTrue(any("A conexão com a IA foi bloqueada" in item.value for item in app.warning))
        technical = "\n".join(item.value for item in app.code)
        self.assertIn("1010", technical)
        self.assertIn("chave oculta", technical)
        rendered = "\n".join(str(item.value) for kind in (app.markdown, app.warning, app.error, app.info, app.code) for item in kind)
        self.assertNotIn(FAKE_SECRET, rendered)
        self.assertNotIn(FAKE_SECRET, feedback["technical_details"])

    def test_loading_new_scenario_clears_previous_report_and_conversation(self):
        app = self.app()
        self.investigate_focus(app)
        self.assertIsNotNone(app.session_state["last_investigation"])
        app.session_state["messages"] = [{"role": "user", "content": "Pergunta sobre a base antiga"}]
        app.selectbox(key="demo_scenario").set_value("Mudança nos descontos").run()
        app.button(key="load_demo").click().run()
        self.assert_no_exceptions(app)
        self.assertIsNone(app.session_state["last_investigation"])
        self.assertEqual(app.session_state["messages"], [])
        self.assertEqual(app.session_state["available_periods"], ["2024-09", "2024-10"])
        self.assertEqual(app.session_state["data_label"], "Mudança nos descontos")

    def test_one_month_has_guidance_instead_of_exception_or_false_alert(self):
        app = self.app()
        data = app.session_state["data"]
        app.session_state["data"] = data.loc[data["data"].dt.month.eq(7)].copy()
        app.session_state["available_periods"] = ["2024-07"]
        app.session_state["data_label"] = "Base de um mês"
        self.navigate(app, "Investigação")
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(any("Precisamos de dois períodos" in item.value for item in app.info))
        self.assertEqual(next(item.value for item in app.metric if item.label == "Variações sinalizadas"), "0")
        self.assertIsNone(app.session_state["last_investigation"])

    def test_chat_investigation_persists_readable_report_without_duplicate_widgets(self):
        app = self.app()
        self.navigate(app, "Assistente")
        app.button(key="chat_suggestion_0").click().run()
        self.assert_no_exceptions(app)
        messages = app.session_state["messages"]
        self.assertEqual(len(messages), 2)
        self.assertIn("report", messages[-1])
        self.assertIn("R$ 41.411,70", messages[-1]["content"])
        app.chat_input[0].set_value("Investigue julho de 2024 e agosto de 2024.").run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.session_state["messages"]), 4)
        app.run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.session_state["messages"]), 4)

    def test_data_view_formats_values_and_offers_report_downloads(self):
        app = self.app()
        self.investigate_focus(app)
        self.navigate(app, "Dados e relatório")
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.dataframe[0].value), 16)
        labels = {item.label for item in app.get("download_button")}
        self.assertIn("Baixar base padronizada", labels)
        self.assertIn("Baixar relatório", labels)
        self.assertIn("Baixar registro técnico", labels)

    def test_gap_between_months_does_not_produce_false_monthly_detection(self):
        app = self.app()
        data = app.session_state["data"].copy()
        august = data["data"].dt.month.eq(8)
        data.loc[august, "data"] += pd.DateOffset(months=1)
        app.session_state["data"] = data
        app.session_state["available_periods"] = ["2024-07", "2024-09"]
        self.navigate(app, "Investigação")
        self.submit(app, "automatic_investigation")
        self.assert_no_exceptions(app)
        self.assertEqual(app.session_state["detection_feedback"]["code"], "missing_consecutive_periods")
        self.assertIsNone(app.session_state["last_investigation"])


if __name__ == "__main__":
    unittest.main()
