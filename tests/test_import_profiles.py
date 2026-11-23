import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data.import_profiles import list_import_profiles, missing_profile_columns, save_import_profile
from src.data.loader import STANDARD_COLUMNS


def profile_config():
    mapping = dict.fromkeys(STANDARD_COLUMNS)
    mapping.update(data="Day", produto="Item", quantidade="Units", preco_unitario="Price", faturamento="Total")
    return {"mapping": mapping, "currency": "GBP", "date_format": "%d/%m/%Y", "decimal": ",", "discount_percent": False,
            "revenue_mode": "column", "price_mode": "column", "invalid_policy": "raise",
            "read_options": {"separator": ";", "encoding": "utf-8-sig", "sheet": None}}


class ImportProfilesTest(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.file = self.root / "input.csv"
        self.file.write_text("Day;Item;Units;Price;Total\n31/01/2024;001;2;10,50;21,00\n", encoding="utf-8")
        setting = patch("src.data.import_profiles.PROFILE_DIR", self.root / "profiles")
        setting.start()
        self.addCleanup(setting.stop)

    def app(self):
        script = f'''
from pathlib import Path
import streamlit as st
from src.ui.import_views import render_import_wizard
def apply(frame, filename):
    st.session_state.applied_frame = frame
    return True
render_import_wizard(Path({str(self.file)!r}).read_bytes(), {self.file.name!r}, apply)
'''
        return AppTest.from_string(script, default_timeout=30).run()

    def check(self, app):
        self.assertEqual([item.message for item in app.exception], [])

    def apply_profile(self, app, identifier):
        next(item for item in app.selectbox if item.label == "Perfil de importação salvo").set_value(identifier).run()
        next(item for item in app.button if item.label == "Aplicar perfil").click().run()
        self.check(app)

    def test_persistence_only_keeps_settings_and_ignores_corrupt_records(self):
        record = save_import_profile("Loja mensal", profile_config())
        restored, invalid = list_import_profiles()
        self.assertEqual(restored, [record])
        self.assertEqual(invalid, 0)
        payload = json.loads((self.root / "profiles" / (record["id"] + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"version", "id", "name", "created_at", "config"})
        (self.root / "profiles" / "bad.json").write_text("{", encoding="utf-8")
        self.assertEqual(list_import_profiles()[1], 1)
        self.assertEqual(missing_profile_columns(record["config"], ["Day", "Item"]), ["Price", "Total", "Units"])

    def test_invalid_profiles_cannot_be_saved(self):
        for key, bad in (("currency", "XXX"), ("invalid_policy", "anything"), ("discount_percent", "false"), ("mapping", {"data": "Day"})):
            config = profile_config()
            config[key] = bad
            with self.assertRaises(ValueError):
                save_import_profile("Inválido", config)
        self.assertFalse((self.root / "profiles").exists())

    def test_profile_reused_in_new_session_requires_validation_before_loading(self):
        record = save_import_profile("Vendas em libras", profile_config())
        app = self.app()
        self.apply_profile(app, record["id"])
        self.assertEqual(next(item.value for item in app.selectbox if item.label == "Moeda dos valores"), "GBP")
        self.assertTrue(next(item for item in app.button if item.label == "Usar esta base").disabled)
        next(item for item in app.button if item.label == "Validar mapeamento").click().run()
        self.check(app)
        prepared = app.session_state["import_prepared"]["data"]
        self.assertEqual(prepared.iloc[0]["produto"], "001")
        self.assertEqual(prepared.iloc[0]["faturamento"], 21.)
        self.assertEqual(str(prepared.iloc[0]["data"].date()), "2024-01-31")
        with self.assertRaises(KeyError):
            app.session_state["applied_frame"]
        next(item for item in app.button if item.label == "Usar esta base").click().run()
        self.check(app)
        self.assertEqual(app.session_state["applied_frame"].attrs["dataset_context"]["currency"], "GBP")
        self.file.write_text("Day;Item;Units;Price;Total\n01/02/2024;002;1;5,00;5,00\n", encoding="utf-8")
        fresh = self.app()
        self.apply_profile(fresh, record["id"])
        next(item for item in fresh.button if item.label == "Validar mapeamento").click().run()
        self.check(fresh)
        self.assertEqual(fresh.session_state["import_prepared"]["data"].iloc[0]["faturamento"], 5.)

    def test_incompatible_columns_block_profile_and_do_not_load_stale_preview(self):
        record = save_import_profile("Loja", profile_config())
        self.file.write_text("Day;Item;Units;Price;Other\n31/01/2024;001;2;10,50;21,00\n", encoding="utf-8")
        app = self.app()
        self.apply_profile(app, record["id"])
        self.assertTrue(any("Total" in item.value for item in app.warning))
        self.assertFalse(any(item.label == "Usar esta base" for item in app.button))
        next(item for item in app.button if item.label == "Descartar perfil e configurar manualmente").click().run()
        self.check(app)
        self.assertTrue(any(item.label == "Validar mapeamento" for item in app.button))

    def test_save_button_uses_last_validated_configuration(self):
        record = save_import_profile("Original", profile_config())
        app = self.app()
        self.apply_profile(app, record["id"])
        next(item for item in app.button if item.label == "Validar mapeamento").click().run()
        self.check(app)
        next(item for item in app.text_input if item.label == "Nome do perfil").set_value("Nova configuração")
        next(item for item in app.button if item.label == "Salvar perfil").click().run()
        self.check(app)
        saved = next(item for item in list_import_profiles()[0] if item["name"] == "Nova configuração")
        self.assertEqual(saved["config"], record["config"])

    def test_excel_profile_uses_saved_sheet_and_blocks_a_missing_sheet(self):
        self.file = self.root / "input.xlsx"
        with pd.ExcelWriter(self.file, engine="openpyxl") as writer:
            pd.DataFrame({"Instructions": ["Select the sales sheet"]}).to_excel(writer, sheet_name="Leia-me", index=False)
            pd.DataFrame({"Day": ["31/01/2024"], "Item": ["A"], "Units": [2], "Price": [10.5], "Total": [21.]}).to_excel(writer, sheet_name="Vendas", index=False)
        config = profile_config()
        config["read_options"]["sheet"] = "Vendas"
        record = save_import_profile("Planilha mensal", config)
        app = self.app()
        self.apply_profile(app, record["id"])
        self.assertEqual(next(item.value for item in app.selectbox if item.label == "Aba da planilha"), "Vendas")
        next(item for item in app.button if item.label == "Validar mapeamento").click().run()
        self.check(app)
        self.assertEqual(app.session_state["import_prepared"]["data"].iloc[0]["faturamento"], 21.)
        pd.DataFrame({"Info": [1]}).to_excel(self.file, sheet_name="Outra", index=False)
        fresh = self.app()
        self.apply_profile(fresh, record["id"])
        self.assertTrue(any("Vendas" in item.value and "não existe" in item.value for item in fresh.warning))
        next(item for item in fresh.button if item.label == "Descartar perfil e configurar manualmente").click().run()
        self.check(fresh)


if __name__ == "__main__":
    unittest.main()
