from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from src.data.adapters.factory import create_sales_data_adapter, load_sales_dataset
from src.data.adapters.online_retail_adapter import OnlineRetailAdapter
from src.data.loader import STANDARD_COLUMNS


def _row(**changes):
    row = {
        "InvoiceNo": "000001", "StockCode": "001A", "Quantity": 2,
        "InvoiceDate": pd.Timestamp("2011-01-02 10:00:00"), "UnitPrice": 3.5,
        "Country": "United Kingdom", "CustomerID": np.nan, "Description": "PRIVATE DESCRIPTION",
    }
    row.update(changes)
    return row


def _xlsx_bytes(rows, other_sheet=False):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Online Retail")
        if other_sheet:
            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="Another sheet")
    return buffer.getvalue()


def _write_xlsx(folder, rows, other_sheet=False):
    path = Path(folder) / "Online Retail.xlsx"
    path.write_bytes(_xlsx_bytes(rows, other_sheet=other_sheet))
    return path


class OnlineRetailAdapterTests(unittest.TestCase):
    def test_xlsx_uses_original_currency_and_unavailable_capabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write_xlsx(folder, [_row()])
            result = OnlineRetailAdapter(path).load()
            self.assertEqual(list(result.columns), STANDARD_COLUMNS)
            self.assertEqual(result.iloc[0]["produto"], "001A")
            self.assertEqual(result.iloc[0]["regiao"], "United Kingdom")
            self.assertEqual(result.iloc[0]["categoria"], "Não informado")
            self.assertEqual(result.iloc[0]["vendedor"], "Não informado")
            self.assertEqual(result.iloc[0]["faturamento"], 7)
            self.assertEqual(result.iloc[0]["preco_unitario"], 3.5)
            self.assertEqual(result.iloc[0]["data"], pd.Timestamp("2011-01-02"))
            context = result.attrs["dataset_context"]
            self.assertEqual(context["currency"], "GBP")
            self.assertEqual(context["currency_symbol"], "£")
            self.assertEqual(context["available_dimensions"], ["produto", "regiao"])
            for field in ("discount_available", "category_available", "seller_available"):
                self.assertFalse(context[field])
            self.assertNotIn("CustomerID", result.columns)
            self.assertNotIn("PRIVATE DESCRIPTION", str(result.attrs))
            self.assertEqual(result.attrs["import_audit"]["retained_source_rows"], 1)

    def test_exclusions_are_disjoint_and_signed_amounts_reconcile(self):
        rows = [
            _row(), _row(InvoiceNo="c00002", Quantity=-2),
            _row(InvoiceNo="000003", Quantity=-1),
            _row(InvoiceNo="000004", Quantity=0),
            _row(InvoiceNo="000005", UnitPrice=0),
            _row(InvoiceNo="000006", UnitPrice=-5),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = OnlineRetailAdapter(_write_xlsx(folder, rows)).load()
            audit = result.attrs["import_audit"]
            self.assertEqual(audit["input_rows"], 6)
            self.assertEqual(audit["excluded_rows"], 5)
            self.assertEqual(audit["retained_source_rows"], 1)
            self.assertEqual([item["rows"] for item in audit["exclusions_ordered"]], [1, 2, 2])
            self.assertEqual(sum(item["rows"] for item in audit["exclusions_ordered"]), 5)
            self.assertEqual(result["faturamento"].sum(), 7)
            self.assertEqual(audit["source_line_value_signed_gbp"], -13.5)
            self.assertEqual(audit["retained_line_value_gbp"] + audit["excluded_line_value_signed_gbp"], -13.5)
            for field in ("rows_match", "quantity_matches", "line_value_matches"):
                self.assertTrue(audit["reconciliation"][field])

    def test_repeated_rows_and_missing_customer_identification_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            result = OnlineRetailAdapter(_write_xlsx(folder, [_row(), _row()])).load()
            self.assertEqual(len(result), 1)
            self.assertEqual(result["quantidade"].sum(), 4)
            self.assertEqual(result["faturamento"].sum(), 14)
            self.assertEqual(result.attrs["import_audit"]["repeated_analysis_rows_retained"], 1)

    def test_aggregated_unit_price_is_weighted_by_quantity(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [_row(Quantity=2, UnitPrice=3), _row(InvoiceNo="00002", Quantity=1, UnitPrice=6)]
            result = OnlineRetailAdapter(_write_xlsx(folder, rows)).load()
            self.assertEqual(result["quantidade"].sum(), 3)
            self.assertEqual(result.iloc[0]["preco_unitario"], 4)
            self.assertEqual(result.iloc[0]["faturamento"], 12)

    def test_country_and_product_missing_values_have_explicit_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            result = OnlineRetailAdapter(_write_xlsx(folder, [_row(StockCode=None, Country=None)])).load()
            self.assertEqual(result.iloc[0]["produto"], "Produto não informado")
            self.assertEqual(result.iloc[0]["regiao"], "País não informado")
            self.assertEqual(result.attrs["import_audit"]["missing_product_rows"], 1)
            self.assertEqual(result.attrs["import_audit"]["missing_country_rows"], 1)

    def test_zip_loads_one_workbook_without_extracting_any_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "online+retail.zip"
            with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr("folder/Online Retail.xlsx", _xlsx_bytes([_row()]))
                archive.writestr("README.txt", "Dataset")
            result = load_sales_dataset(dataset_type="online_retail", input_path=path)
            self.assertEqual(result["faturamento"].sum(), 7)
            self.assertEqual(result.attrs["import_audit"]["archive_member"], "folder/Online Retail.xlsx")
            self.assertEqual([entry.name for entry in Path(folder).iterdir()], [path.name])

    def test_missing_required_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            row = _row()
            del row["UnitPrice"]
            with self.assertRaisesRegex(ValueError, "UnitPrice"):
                OnlineRetailAdapter(_write_xlsx(folder, [row])).load()

    def test_invalid_dates_and_numeric_values_are_rejected_instead_of_dropped(self):
        bad_changes = [
            {"InvoiceDate": "not a date"}, {"InvoiceDate": None}, {"InvoiceDate": 12345},
            {"Quantity": "invalid"}, {"UnitPrice": np.inf}, {"UnitPrice": None},
            {"InvoiceNo": None},
        ]
        for changes in bad_changes:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    OnlineRetailAdapter(_write_xlsx(folder, [_row(), _row(**changes)])).load()

    def test_text_dates_are_interpreted_with_day_first(self):
        with tempfile.TemporaryDirectory() as folder:
            result = OnlineRetailAdapter(_write_xlsx(folder, [_row(InvoiceDate="02/01/2011 10:00")])).load()
            self.assertEqual(result.iloc[0]["data"], pd.Timestamp("2011-01-02"))

    def test_multiple_sheets_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = _write_xlsx(folder, [_row()], other_sheet=True)
            with self.assertRaisesRegex(ValueError, "várias abas"):
                OnlineRetailAdapter(path).load()
            result = OnlineRetailAdapter(path, sheet_name="Online Retail").load()
            self.assertEqual(result.attrs["import_audit"]["sheet"], "Online Retail")
            with self.assertRaisesRegex(ValueError, "não existe"):
                OnlineRetailAdapter(path, sheet_name="Missing").load()

    def test_unsafe_zip_names_and_links_are_rejected(self):
        for name in ("../Online Retail.xlsx", "/Online Retail.xlsx", "C:/Online Retail.xlsx", "..\\Online Retail.xlsx"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "unsafe.zip"
                with ZipFile(path, "w") as archive:
                    archive.writestr(name, b"not read")
                with self.assertRaisesRegex(ValueError, "insegura"):
                    OnlineRetailAdapter(path).load()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "link.zip"
            link = ZipInfo("Online Retail.xlsx")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            with ZipFile(path, "w") as archive:
                archive.writestr(link, "target")
            with self.assertRaisesRegex(ValueError, "insegura"):
                OnlineRetailAdapter(path).load()

    def test_ambiguous_archive_and_expansion_limits_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "multiple.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("one.xlsx", b"ignored")
                archive.writestr("two.xlsx", b"ignored")
            with self.assertRaisesRegex(ValueError, "exatamente uma"):
                OnlineRetailAdapter(path).load()
            path = Path(folder) / "too_big.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("Online Retail.xlsx", b"a" * 20)
            adapter = OnlineRetailAdapter(path)
            adapter.MAX_OUTER_UNCOMPRESSED_BYTES = 10
            with self.assertRaisesRegex(ValueError, "descompactado"):
                adapter.load()

    def test_empty_scope_missing_file_and_unsupported_extension_fail_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Nenhuma venda"):
                OnlineRetailAdapter(_write_xlsx(folder, [_row(Quantity=-1)])).load()
            with self.assertRaises(FileNotFoundError):
                OnlineRetailAdapter(Path(folder) / "missing.xlsx").load()
            other = Path(folder) / "retail.csv"
            other.write_text("data", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "xlsx"):
                OnlineRetailAdapter(other).load()
            with self.assertRaisesRegex(ValueError, "input_path"):
                create_sales_data_adapter(dataset_type="online_retail")


if __name__ == "__main__":
    unittest.main()
