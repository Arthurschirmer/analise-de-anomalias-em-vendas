"""Real Olist integration checks, skipped when the local public dataset is absent.

These tests exercise local adaptation and UI flow. They do not estimate detection
accuracy or causal correctness, and cannot call an external model service.
"""

import math
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.llm.mock_provider import MockLLMProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "olist_raw"
REQUIRED_FILES = [
    RAW_DIR / filename
    for filename in (
        "olist_orders_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_products_dataset.csv",
        "olist_customers_dataset.csv",
    )
]


@unittest.skipUnless(all(path.is_file() for path in REQUIRED_FILES), "Local Olist CSV files not available")
class OlistWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orders = pd.read_csv(
            RAW_DIR / "olist_orders_dataset.csv",
            usecols=["order_id", "order_status", "order_purchase_timestamp"],
        )
        items = pd.read_csv(
            RAW_DIR / "olist_order_items_dataset.csv",
            usecols=["order_id", "price"],
        )
        eligible = items.merge(
            orders.loc[orders["order_status"].eq("delivered")],
            on="order_id",
            how="inner",
            validate="many_to_one",
        )
        dates = pd.to_datetime(eligible["order_purchase_timestamp"]).dt.floor("D")
        cls.expected_items = len(eligible)
        cls.expected_revenue = float(eligible["price"].sum())
        cls.expected_start = dates.min()
        cls.expected_end = dates.max()
        monthly = eligible.assign(period=dates.dt.to_period("M").astype(str))
        cls.expected_monthly = monthly.groupby("period")["price"].sum()

    def setUp(self):
        provider_patch = patch(
            "src.ui.workspace.create_llm_provider_from_env",
            return_value=MockLLMProvider(),
        )
        provider_patch.start()
        self.addCleanup(provider_patch.stop)
        network_patch = patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("External API calls are disabled in Olist integration tests"),
        )
        network_patch.start()
        self.addCleanup(network_patch.stop)

    def load_app(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
        app.selectbox(key="data_source_selector").set_value("Olist local").run()
        app.button(key="load_olist").click().run()
        self.assert_no_exceptions(app)
        self.assertIsNone(app.session_state["load_feedback"])
        return app

    def assert_no_exceptions(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def test_real_olist_dashboard_reconciles_items_revenue_and_months(self):
        app = self.load_app()
        data = app.session_state["data"]
        self.assertEqual(app.session_state["data_source_type"], "olist")
        self.assertGreater(len(data), 0)
        self.assertEqual(int(data["quantidade"].sum()), self.expected_items)
        self.assertAlmostEqual(float(data["faturamento"].sum()), self.expected_revenue, places=6)
        self.assertEqual(data["data"].min(), self.expected_start)
        self.assertEqual(data["data"].max(), self.expected_end)
        monthly = data.groupby(data["data"].dt.to_period("M").astype(str))["faturamento"].sum()
        self.assertEqual(monthly.index.tolist(), self.expected_monthly.index.tolist())
        for month, revenue in self.expected_monthly.items():
            with self.subTest(month=month):
                self.assertAlmostEqual(float(monthly.loc[month]), float(revenue), places=6)
        numeric = data[["quantidade", "faturamento", "preco_unitario", "desconto"]]
        self.assertTrue(numeric.apply(lambda column: column.map(math.isfinite)).all().all())
        self.assertEqual(len(app.error), 0)

    def test_real_olist_investigation_and_report_export_complete_offline(self):
        app = self.load_app()
        app.session_state["workspace_view"] = "Investigação"
        app.run()
        submit = next(
            item for item in app.button
            if (item.key or "").startswith("FormSubmitter:automatic_investigation-")
        )
        submit.click().run()
        self.assert_no_exceptions(app)
        self.assertIsNone(app.session_state["detection_feedback"])
        self.assertEqual(app.session_state["last_investigation_mode"], "local")
        investigation = app.session_state["last_investigation"]
        self.assertIsNotNone(investigation)
        report = investigation.structured_report
        self.assertFalse(report["contexto_da_base"]["discount_available"])
        self.assertIn("Não disponível", [item.value for item in app.metric if item.label == "Mudança no desconto médio"])
        periods = report["periodos"]
        data = app.session_state["data"]
        month = data["data"].dt.to_period("M").astype(str)
        previous = float(data.loc[month.eq(periods["periodo_anterior"]), "faturamento"].sum())
        current = float(data.loc[month.eq(periods["periodo_atual"]), "faturamento"].sum())
        comparison = report["resumo_comparativo"]
        self.assertAlmostEqual(comparison["faturamento_anterior"], previous, places=6)
        self.assertAlmostEqual(comparison["faturamento_atual"], current, places=6)
        self.assertAlmostEqual(comparison["variacao_faturamento"], current - previous, places=6)
        self.assertTrue(report["causas_provaveis"])

        app.session_state["workspace_view"] = "Dados e relatório"
        app.run()
        self.assert_no_exceptions(app)
        self.assertEqual(len(app.dataframe[0].value), min(100, len(data)))
        download_labels = {item.label for item in app.get("download_button")}
        self.assertTrue(
            {"Baixar base padronizada", "Baixar relatório", "Baixar registro técnico"}.issubset(download_labels)
        )

    def test_explicit_real_data_window_and_unknown_discount_are_preserved(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
        app.selectbox(key="data_source_selector").set_value("Olist local").run()
        app.checkbox(key="olist_use_range").check().run()
        app.button(key="load_olist").click().run()
        self.assert_no_exceptions(app)
        self.assertIsNone(app.session_state["load_feedback"])
        data = app.session_state["data"]
        periods = data["data"].dt.to_period("M").astype(str)
        self.assertEqual(periods.min(), "2017-01")
        self.assertEqual(periods.max(), "2018-07")
        expected = float(self.expected_monthly.loc["2017-01":"2018-07"].sum())
        self.assertAlmostEqual(float(data["faturamento"].sum()), expected, places=6)
        self.assertEqual(data.attrs["dataset_context"]["selected_start"], "2017-01")
        self.assertFalse(data.attrs["dataset_context"]["discount_available"])
        app.session_state["workspace_view"] = "Assistente"
        app.run()
        app.chat_input[0].set_value("Qual foi o desconto médio?").run()
        self.assert_no_exceptions(app)
        answer = app.session_state["messages"][-1]["content"]
        self.assertIn("não fornece o desconto", answer)
        self.assertIn("preenchido", answer)


if __name__ == "__main__":
    unittest.main()

