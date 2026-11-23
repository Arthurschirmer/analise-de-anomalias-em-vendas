"""Exercise the UCI source through the interface without a remote model."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider
from src.ui.dataset_context import dataset_context
from src.data.adapters import load_sales_dataset
from src.ui.workspace import _local_conclusion, _report_markdown


ROOT = Path(__file__).resolve().parents[1]


class OnlineRetailWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = TemporaryDirectory()
        cls.file = Path(cls.folder.name) / "retail.xlsx"
        rows = []
        for index, date in enumerate(pd.date_range("2011-01-01", periods=140)):
            for code, country in (("001A", "United Kingdom"), ("002B", "France")):
                quantity = 2 if index == 130 else 10
                rows.append({"InvoiceNo": str(index + 1000), "StockCode": code,
                             "InvoiceDate": date, "Quantity": quantity,
                             "UnitPrice": 12.5, "Country": country})
        pd.DataFrame(rows).to_excel(cls.file, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        provider = patch("src.ui.workspace.create_llm_provider_from_env", return_value=MockLLMProvider())
        provider.start()
        self.addCleanup(provider.stop)
        network = patch("urllib.request.urlopen", side_effect=AssertionError("No remote calls in UCI tests"))
        network.start()
        self.addCleanup(network.stop)

    def load_app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        app.selectbox(key="data_source_selector").set_value("UCI Online Retail").run()
        app.text_input(key="online_retail_path").set_value(str(self.file)).run()
        app.button(key="load_online_retail").click().run()
        self.assertEqual([item.message for item in app.exception], [])
        self.assertIsNone(app.session_state["load_feedback"])
        return app

    def test_currency_and_dimensions_are_kept_in_dashboard_and_history(self):
        app = self.load_app()
        self.assertEqual(app.session_state["data_source_type"], "online_retail")
        self.assertTrue(next(item.value for item in app.metric if item.label == "Faturamento da base").startswith("£ "))
        labels = app.segmented_control(key="dashboard_dimension").options
        self.assertEqual(labels, ["Código do produto", "País do cliente"])
        app.session_state["workspace_view"] = "Histórico"
        app.run()
        self.assertEqual(app.selectbox(key="historical_scope").options, ["Base inteira", "Por país"])
        submit = next(item for item in app.button if (item.key or "").startswith("FormSubmitter:historical_detection-"))
        submit.click().run()
        self.assertIsNotNone(app.session_state["historical_result"])
        self.assertEqual([item.message for item in app.exception], [])

    def test_monthly_investigation_and_exports_keep_gbp_and_unavailable_fields(self):
        app = self.load_app()
        app.session_state["workspace_view"] = "Investigação"
        app.run()
        app.segmented_control(key="investigation_kind").set_value("Comparar períodos").run()
        app.selectbox(key="compare_previous").set_value("2011-03")
        app.selectbox(key="compare_current").set_value("2011-04")
        button = next(item for item in app.button if (item.key or "").startswith("FormSubmitter:manual_investigation-"))
        button.click().run()
        self.assertEqual([item.message for item in app.exception], [])
        report = app.session_state["last_investigation"].structured_report
        self.assertEqual(report["contexto_da_base"]["currency"], "GBP")
        self.assertTrue(all(cause["dimensao"] in {"produto", "regiao"} for cause in report["causas_provaveis"]))
        self.assertIn("Não disponível", [item.value for item in app.metric if item.label == "Mudança no desconto médio"])
        exported = _report_markdown(app.session_state["last_investigation"], "local")
        self.assertIn("£", exported)
        self.assertNotIn("R$", exported)
        app.session_state["workspace_view"] = "Dados e relatório"
        app.run()
        self.assertEqual([item.message for item in app.exception], [])
        self.assertIn("Baixar definições e auditoria", [item.label for item in app.get("download_button")])

    def test_chat_does_not_treat_placeholders_as_known_information(self):
        app = self.load_app()
        app.session_state["workspace_view"] = "Assistente"
        app.run()
        for question, expected in (("Qual foi o desconto médio?", "não fornece o desconto"),
                                   ("Por que caiu a categoria?", "não informa categoria"),
                                   ("Qual vendedor mais vendeu?", "não informa vendedor")):
            app.chat_input[0].set_value(question).run()
            answer = app.session_state["messages"][-1]["content"]
            self.assertIn(expected, answer)
            self.assertNotIn("base Olist", answer)
            self.assertEqual([item.message for item in app.exception], [])
        app.chat_input[0].set_value("Qual foi o faturamento total?").run()
        self.assertIn("£", app.session_state["messages"][-1]["content"])

    def test_dataset_context_explains_the_source_without_inventing_categories(self):
        data = load_sales_dataset("online_retail", input_path=self.file)
        context = dataset_context(data, "online_retail")
        self.assertEqual(context["currency"], "GBP")
        self.assertFalse(context["category_available"])
        self.assertFalse(context["seller_available"])
        self.assertFalse(context["discount_available"])
        self.assertEqual(context["available_dimensions"], ["produto", "regiao"])
        self.assertEqual(context["missing_category"]["row_count"], len(data))
        self.assertIn("sem conversão", " ".join(context["notes"]))

    def test_saved_report_format_does_not_depend_on_the_current_session_currency(self):
        report = {"contexto_da_base": {"currency": "GBP"},
                  "resumo_comparativo": {"faturamento_anterior": 100, "faturamento_atual": 125,
                                         "variacao_faturamento": 25, "variacao_percentual_faturamento": 25},
                  "periodos": {"periodo_anterior": "2011-01", "periodo_atual": "2011-02"},
                  "causas_provaveis": []}
        text = _local_conclusion(report)
        self.assertIn("£", text)
        self.assertNotIn("R$", text)


if __name__ == "__main__":
    unittest.main()
