from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider


class ChatComparisonContextTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.csv = self.root / "sales.csv"
        pd.DataFrame({"data": ["2024-01-10", "2024-01-10", "2024-02-10", "2024-02-10", "2024-03-10", "2024-03-10"],
            "produto": ["A", "B"] * 3, "categoria": "C", "regiao": "UK", "vendedor": "V",
            "quantidade": [10, 20, 5, 20, 4, 20], "preco_unitario": [100, 10] * 3,
            "desconto": 0., "faturamento": [1000, 200, 500, 200, 400, 200]}).to_csv(self.csv, index=False)
        for context in (patch("src.ui.workspace.create_llm_provider_from_env", return_value=MockLLMProvider()),
                        patch("src.ui.workspace.HISTORY_DIR", self.root / "history"),
                        patch("urllib.request.urlopen", side_effect=AssertionError("No API calls in context checks"))):
            context.start()
            self.addCleanup(context.stop)

    def app(self):
        source = f'''
import streamlit as st
from src.ui.workspace import main, _init_state, _set_dataset
from src.data.loader import load_sales_csv
_init_state()
if not st.session_state.get("fixture_loaded"):
    data = load_sales_csv({str(self.csv)!r})
    data.attrs["dataset_context"] = {{"currency": "GBP", "source_label": "Teste em libras"}}
    _set_dataset(data, "fixture.csv", "Teste em libras", "csv")
    st.session_state.fixture_loaded = True
    st.session_state.workspace_view = "Assistente"
main()
'''
        result = AppTest.from_string(source, default_timeout=30).run()
        self.check(result)
        return result

    def check(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def ask(self, app, question):
        app.chat_input(key="chat_prompt").set_value(question).run()
        self.check(app)
        return app.session_state["messages"][-1]

    def test_followup_reuses_report_without_api_recalculation_or_duplicate_history(self):
        app = self.app()
        first = self.ask(app, "Investigue janeiro de 2024 e fevereiro de 2024.")
        self.assertEqual(first["report"]["resumo_comparativo"]["faturamento_anterior"], 1200)
        self.assertEqual(first["report"]["resumo_comparativo"]["faturamento_atual"], 700)
        with patch("src.agent.investigation_agent.InvestigationAgent.investigate", side_effect=AssertionError("Reuse calculated evidence")):
            followup = self.ask(app, "E por que isso ocorreu?")
        self.assertEqual(followup["report"], first["report"])
        self.assertIn("£ 1.200,00", followup["content"])
        self.assertIn("£ 700,00", followup["content"])
        self.assertIn("precisa de confirmação", followup["content"])
        self.assertEqual(len(list((self.root / "history").glob("*.json"))), 1)
        self.assertTrue(any("jan/2024" in item.value and "fev/2024" in item.value for item in app.info))

    def test_one_named_month_uses_previous_month_and_states_reference(self):
        app = self.app()
        answer = self.ask(app, "Por que as vendas caíram em fevereiro de 2024?")
        self.assertIn("Usei **jan/2024** como referência", answer["content"])
        self.assertEqual(answer["report"]["periodos"], {"periodo_anterior": "2024-01", "periodo_atual": "2024-02"})

    def test_new_explicit_pair_updates_context(self):
        app = self.app()
        self.ask(app, "Investigue janeiro de 2024 e fevereiro de 2024.")
        answer = self.ask(app, "Investigue fevereiro de 2024 e março de 2024.")
        self.assertEqual(answer["report"]["periodos"]["periodo_atual"], "2024-03")
        followup = self.ask(app, "E por que isso ocorreu?")
        self.assertEqual(followup["report"]["resumo_comparativo"]["faturamento_anterior"], 700)
        self.assertEqual(followup["report"]["resumo_comparativo"]["faturamento_atual"], 600)

    def test_followup_after_a_regular_comparison_uses_that_pair(self):
        app = self.app()
        self.ask(app, "Compare janeiro de 2024 e fevereiro de 2024.")
        answer = self.ask(app, "E por que isso ocorreu?")
        self.assertEqual(answer["report"]["periodos"], {"periodo_anterior": "2024-01", "periodo_atual": "2024-02"})
        self.assertEqual(answer["report"]["resumo_comparativo"]["faturamento_atual"], 700)

    def test_unknown_reference_requires_a_choice_and_selector_can_start_investigation(self):
        app = self.app()
        answer = self.ask(app, "E por que isso ocorreu?")
        self.assertNotIn("report", answer)
        self.assertIn("Escolha uma variação", answer["content"])
        app.selectbox(key="chat_monthly_selection").set_value("2024-02").run()
        app.button(key="chat_investigate_selection").click().run()
        self.check(app)
        self.assertEqual(app.session_state["last_investigation"].structured_report["periodos"]["periodo_atual"], "2024-02")

    def test_clear_chat_and_dataset_change_discard_monthly_context(self):
        app = self.app()
        self.ask(app, "Investigue janeiro de 2024 e fevereiro de 2024.")
        app.button(key="clear_chat").click().run()
        self.check(app)
        self.assertIsNone(app.session_state["monthly_chat_context"])
        self.assertNotIn("report", self.ask(app, "E por que isso ocorreu?"))
        self.ask(app, "Investigue janeiro de 2024 e fevereiro de 2024.")
        app.button(key="exit_monthly_chat").click().run()
        self.check(app)
        self.assertIsNone(app.session_state["monthly_chat_context"])
        self.ask(app, "Investigue janeiro de 2024 e fevereiro de 2024.")
        app.button(key="load_demo").click().run()
        self.check(app)
        self.assertIsNone(app.session_state["monthly_chat_context"])
        self.assertEqual(app.session_state["messages"], [])

    def test_analysis_result_opens_chat_with_same_comparison(self):
        app = self.app()
        self.ask(app, "Investigue janeiro de 2024 e fevereiro de 2024.")
        app.session_state["workspace_view"] = "Investigação"
        app.run()
        self.check(app)
        app.button(key="chat_about_investigation").click().run()
        self.check(app)
        self.assertEqual(app.session_state["workspace_view"], "Assistente")
        self.assertEqual(app.session_state["messages"], [])
        followup = self.ask(app, "Por que isso ocorreu?")
        self.assertEqual(followup["report"]["periodos"]["periodo_atual"], "2024-02")


if __name__ == "__main__":
    unittest.main()
