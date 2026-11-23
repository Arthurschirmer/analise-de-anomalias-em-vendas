from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider


class ImprovedWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.file = self.root / "sales.csv"
        dates = pd.date_range("2011-04-01", "2011-08-09")
        self.data = pd.DataFrame({"data": dates, "produto": "A", "categoria": "C", "regiao": "Sul",
            "vendedor": "V", "quantidade": 2, "preco_unitario": 10., "desconto": 0., "faturamento": 20.})
        self.data.to_csv(self.file, index=False)
        for context in (
            patch("src.ui.workspace.create_llm_provider_from_env", return_value=MockLLMProvider()),
            patch("src.ui.workspace.HISTORY_DIR", self.root / "history"),
            patch("urllib.request.urlopen", side_effect=AssertionError("No API calls in workflow checks")),
        ):
            context.start()
            self.addCleanup(context.stop)

    def app(self):
        source = f'''
import streamlit as st
from src.ui.workspace import main, _init_state, _set_dataset
from src.data.loader import load_sales_csv
_init_state()
if not st.session_state.get("fixture_loaded"):
    _set_dataset(load_sales_csv({str(self.file)!r}), "fixture.csv", "Base de teste", "csv")
    st.session_state.fixture_loaded = True
main()
'''
        return AppTest.from_string(source, default_timeout=45).run()

    def compare(self, app, mode):
        app.session_state["workspace_view"] = "Investigação"
        app.run()
        app.segmented_control(key="investigation_kind").set_value("Comparar períodos").run()
        app.selectbox(key="compare_previous").set_value("2011-07")
        app.selectbox(key="compare_current").set_value("2011-08")
        app.selectbox(key="comparison_window_mode").set_value(mode)
        next(button for button in app.button if (button.key or "").startswith("FormSubmitter:manual_investigation-")).click().run()
        self.assertEqual([item.message for item in app.exception], [])
        self.assertTrue(any(item.label == "Baixar investigação em PDF" for item in app.get("download_button")))
        return app.session_state["last_investigation"]

    def test_aligned_comparison_and_reopen_preserve_results_without_running_agent(self):
        app = self.app()
        log = self.compare(app, "aligned_days")
        summary = log.structured_report["resumo_comparativo"]
        self.assertEqual(summary["faturamento_anterior"], 180)
        self.assertEqual(summary["faturamento_atual"], 180)
        self.assertEqual(summary["variacao_faturamento"], 0)
        self.assertEqual(len(list((self.root / "history").glob("*.json"))), 1)
        self.assertTrue(any("dias 1 a 9" in item.value for item in app.info))
        original_id = app.session_state["last_saved_id"]
        app.session_state["workspace_view"] = "Dados e relatório"
        app.session_state["last_investigation"] = None
        app.run()
        self.assertEqual([item.message for item in app.exception], [])
        with patch("src.agent.investigation_agent.InvestigationAgent.investigate", side_effect=AssertionError("Must not recalculate")):
            app.button(key="history_restore").click().run()
        restored = app.session_state["last_investigation"]
        self.assertEqual(restored.created_at, log.created_at)
        self.assertEqual(restored.structured_report, log.structured_report)
        self.assertEqual(app.session_state["last_saved_id"], original_id)
        self.assertEqual([item.message for item in app.exception], [])
        app.button(key="chat_about_investigation").click().run()
        with patch("src.agent.investigation_agent.InvestigationAgent.investigate", side_effect=AssertionError("Keep the saved day window")):
            app.chat_input(key="chat_prompt").set_value("E por que isso ocorreu?").run()
        self.assertEqual([item.message for item in app.exception], [])
        reply = app.session_state["messages"][-1]
        self.assertEqual(reply["report"]["resumo_comparativo"]["faturamento_anterior"], 180)
        self.assertEqual(reply["report"]["resumo_comparativo"]["faturamento_atual"], 180)
        self.assertIn("dias 1 a 9", reply["content"])

    def test_two_saved_comparisons_can_be_inspected_side_by_side(self):
        app = self.app()
        self.compare(app, "full")
        self.compare(app, "aligned_days")
        app.session_state["workspace_view"] = "Dados e relatório"
        app.run()
        options = app.multiselect(key="history_compare").options
        self.assertEqual(len(options), 2)
        # Values are identifiers, while the rendered options are friendly labels.
        ids = [path.stem for path in (self.root / "history").glob("*.json")]
        app.multiselect(key="history_compare").set_value(ids).run()
        self.assertEqual([item.message for item in app.exception], [])
        frame = next(item.value for item in app.dataframe if "Indicador" in item.value.columns)
        values = set(frame.loc[frame["Indicador"].eq("Diferença de faturamento"), ["Registro 1", "Registro 2"]].iloc[0])
        self.assertEqual(values, {"R$ -440,00", "R$ 0,00"})

    def test_historical_alert_has_a_readable_card_with_known_numbers(self):
        changed = self.data.copy()
        mask = changed["data"].eq(pd.Timestamp("2011-08-05"))
        changed.loc[mask, "quantidade"] = 1
        changed.loc[mask, "faturamento"] = 10
        changed.to_csv(self.file, index=False)
        app = self.app()
        app.session_state["workspace_view"] = "Histórico"
        app.run()
        next(button for button in app.button if (button.key or "").startswith("FormSubmitter:historical_detection-")).click().run()
        self.assertEqual([item.message for item in app.exception], [])
        app.selectbox(key="historical_alert_date").set_value("2011-08-05").run()
        self.assertEqual(next(item.value for item in app.metric if item.label == "Valor observado"), "R$ 10,00")
        self.assertEqual(next(item.value for item in app.metric if item.label == "Referência esperada"), "R$ 20,00")
        self.assertTrue(any("Baixar ficha do alerta" == item.label for item in app.get("download_button")))
        self.assertTrue(any("não é uma probabilidade" in item.value for item in app.caption))
        self.assertTrue(any(item.label == "Baixar alerta em PDF" for item in app.get("download_button")))
        app.button(key="chat_about_alert").click().run()
        self.assertEqual([item.message for item in app.exception], [])
        self.assertEqual(app.session_state["workspace_view"], "Assistente")
        self.assertEqual(app.session_state["alert_chat_context"]["period"], "2011-08-05")
        with patch("src.ui.workspace._provider_for_request", side_effect=AssertionError("Alert answers must stay local")):
            app.button(key="chat_suggestion_0").click().run()
            self.assertEqual([item.message for item in app.exception], [])
            self.assertIn("R$ 10,00", app.session_state["messages"][-1]["content"])
            self.assertNotIn("preciso de dois períodos", app.session_state["messages"][-1]["content"])
            app.chat_input(key="chat_prompt").set_value("Qual foi a quantidade?").run()
            self.assertEqual([item.message for item in app.exception], [])
            self.assertIn("1,00 unidades", app.session_state["messages"][-1]["content"])
        app.button(key="exit_alert_chat").click().run()
        self.assertIsNone(app.session_state["alert_chat_context"])
        self.assertEqual(app.session_state["messages"], [])
        app.session_state["alert_chat_context"] = {"period": "2011-08-05"}
        app.button(key="load_demo").click().run()
        self.assertEqual([item.message for item in app.exception], [])
        self.assertIsNone(app.session_state["alert_chat_context"])


class GuidedImportViewsTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.file = Path(self.folder.name, "input.csv")
        self.file.write_text("Date,Product,Quantity,UnitPrice,Revenue\n2024-01-01,A,2,10,20\n2024-01-02,B,3,10,30\n", encoding="utf-8")

    def app(self):
        script = f'''
from pathlib import Path
import streamlit as st
from src.ui.import_views import render_import_wizard
def apply(frame, filename):
    st.session_state.applied_frame = frame
    return True
if st.session_state.get("import_applied"):
    st.success("Base carregada")
else:
    render_import_wizard(Path({str(self.file)!r}).read_bytes(), "input.csv", apply)
'''
        return AppTest.from_string(script, default_timeout=30).run()

    def select(self, app, suffix, value):
        next(item for item in app.selectbox if (item.key or "").endswith(suffix)).set_value(value)

    def validate(self, app):
        next(item for item in app.button if item.label == "Validar mapeamento").click().run()
        self.assertEqual([item.message for item in app.exception], [])

    def test_preview_then_apply_revalidates_current_form_choices(self):
        app = self.app()
        self.select(app, "_currency", "GBP")
        self.validate(app)
        self.assertEqual(len(app.session_state["import_prepared"]["data"]), 2)
        # Changing the form before apply must not load the older prepared value.
        self.select(app, "_currency", "USD")
        next(item for item in app.button if item.label == "Usar esta base").click().run()
        data = app.session_state["applied_frame"]
        self.assertEqual(data.attrs["dataset_context"]["currency"], "USD")
        self.assertFalse(data.attrs["dataset_context"]["discount_available"])
        self.assertTrue(app.success)

    def test_invalid_rows_are_only_excluded_when_the_policy_is_selected(self):
        self.file.write_text("Date,Product,Quantity,UnitPrice,Revenue\n2024-01-01,A,2,10,20\nbad,B,3,10,30\n", encoding="utf-8")
        app = self.app()
        self.validate(app)
        self.assertEqual(app.session_state["import_validation"]["invalid_rows"], 1)
        self.assertTrue(next(item for item in app.button if item.label == "Usar esta base").disabled)
        self.select(app, "_policy", "drop")
        self.validate(app)
        self.assertEqual(len(app.session_state["import_prepared"]["data"]), 1)
        next(item for item in app.button if item.label == "Usar esta base").click().run()
        self.assertEqual(app.session_state["applied_frame"].attrs["import_validation_report"]["invalid_rows"], 1)


if __name__ == "__main__":
    unittest.main()
