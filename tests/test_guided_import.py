from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data.importing import read_tabular_file, suggest_mapping, map_sales_data
from src.data.loader import prepare_sales_data, load_sales_csv, SalesValidationError
from src.data.capabilities import available_dimensions


def standard_frame():
    return pd.DataFrame({"data": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "produto": ["A", "B", "C"], "categoria": ["C"] * 3, "regiao": ["Sul"] * 3,
        "vendedor": ["V"] * 3, "quantidade": [1, 2, 3], "preco_unitario": [10., 10., 10.],
        "desconto": [0.] * 3, "faturamento": [10., 20., 30.]})


class GuidedImportTest(unittest.TestCase):
    def test_cli_and_prepare_share_invalid_date_and_infinite_number_decisions(self):
        raw = standard_frame()
        raw.loc[0, "data"] = "invalida"
        raw.loc[1, "faturamento"] = float("inf")
        with self.assertRaises(SalesValidationError) as direct:
            prepare_sales_data(raw)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            raw.to_csv(path, index=False)
            with self.assertRaises(SalesValidationError) as loaded:
                load_sales_csv(path)
        self.assertEqual(direct.exception.report, loaded.exception.report)
        self.assertEqual(direct.exception.report["invalid_rows"], 2)
        self.assertEqual(direct.exception.report["loaded_rows"], 0)
        self.assertEqual([issue["source_rows_sample"] for issue in direct.exception.report["issues"]], [[2], [3]])

    def test_explicit_exclusion_keeps_audit_and_unchanged_input(self):
        raw = standard_frame()
        raw.loc[1, "data"] = None
        filtered = prepare_sales_data(raw, invalid_policy="drop")
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.attrs["import_validation_report"]["invalid_rows"], 1)
        revalidated = prepare_sales_data(filtered)
        self.assertEqual(revalidated.attrs["import_validation_report"]["input_rows"], 3)
        self.assertTrue(pd.isna(raw.loc[1, "data"]))
        self.assertEqual(raw.loc[0, "data"], "2024-01-01")

    def test_negative_business_values_are_not_silently_removed(self):
        raw = standard_frame()
        raw.loc[0, "quantidade"] = -1
        raw.loc[0, "faturamento"] = -10
        result = prepare_sales_data(raw)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[0, "faturamento"], -10)
        self.assertTrue(result.attrs["validation_report"]["warnings"])

    def test_csv_mapping_decimal_comma_ids_and_missing_dimensions(self):
        content = b"Date;Product;Quantity;UnitPrice;Revenue\n31/01/2024;0012;2;1.234,50;2.469,00\n"
        raw, _ = read_tabular_file(content, "vendas.csv", separator=";")
        data = map_sales_data(raw, suggest_mapping(raw.columns), currency="GBP", date_format="%d/%m/%Y", decimal=",")
        self.assertEqual(data.loc[0, "produto"], "0012")
        self.assertEqual(data.loc[0, "preco_unitario"], 1234.5)
        self.assertEqual(data.loc[0, "faturamento"], 2469)
        self.assertEqual(data.loc[0, "data"], pd.Timestamp("2024-01-31"))
        self.assertEqual(available_dimensions(data), ("produto",))
        self.assertFalse(data.attrs["dataset_context"]["discount_available"])
        self.assertEqual(data.attrs["dataset_context"]["currency"], "GBP")
        self.assertEqual(data.attrs["mapping_audit"]["read_options"]["separator"], ";")

    def test_xlsx_sheet_selection_preserves_typed_dates_and_numeric_values(self):
        stream = BytesIO()
        raw = pd.DataFrame({"Date": [pd.Timestamp("2024-02-01")], "Product": ["0001"],
                            "Quantity": [2], "UnitPrice": [12.5], "Revenue": [25.]})
        with pd.ExcelWriter(stream, engine="openpyxl") as workbook:
            pd.DataFrame({"nota": ["outra aba"]}).to_excel(workbook, sheet_name="Notas", index=False)
            raw.to_excel(workbook, sheet_name="Vendas", index=False)
        content = stream.getvalue()
        selected, names = read_tabular_file(content, "vendas.xlsx", sheet="Vendas")
        self.assertEqual(names, ["Notas", "Vendas"])
        data = map_sales_data(selected, suggest_mapping(selected.columns), decimal=",")
        self.assertEqual(data.loc[0, "preco_unitario"], 12.5)
        self.assertEqual(data.loc[0, "produto"], "0001")
        self.assertEqual(data.attrs["mapping_audit"]["read_options"]["sheet"], "Vendas")

    def test_derived_values_and_percent_discount_require_explicit_definitions(self):
        raw = pd.DataFrame({"Date": ["2024-01-01"], "Product": ["A"], "Quantity": ["2"],
                            "UnitPrice": ["100"], "Discount": ["10%"]})
        data = map_sales_data(raw, suggest_mapping(raw.columns), revenue_mode="calculate", discount_percent=True)
        self.assertEqual(data.loc[0, "faturamento"], 180)
        self.assertEqual(data.loc[0, "desconto"], .1)
        no_discount = raw.drop(columns="Discount")
        data = map_sales_data(no_discount, suggest_mapping(no_discount.columns), revenue_mode="calculate")
        self.assertEqual(data.loc[0, "faturamento"], 200)
        self.assertFalse(data.attrs["dataset_context"]["discount_available"])
        self.assertIn("hipótese explícita", " ".join(data.attrs["dataset_context"]["limitations"]))

    def test_derived_unit_price_zero_quantity_is_invalid(self):
        raw = pd.DataFrame({"Date": ["2024-01-01"], "Product": ["A"], "Quantity": [0], "Revenue": [100.]})
        with self.assertRaises(SalesValidationError):
            map_sales_data(raw, suggest_mapping(raw.columns), price_mode="calculate")

    def test_mapping_rejects_circular_calculation_and_duplicate_columns(self):
        raw = standard_frame()
        with self.assertRaisesRegex(ValueError, "depender"):
            map_sales_data(raw, suggest_mapping(raw.columns), revenue_mode="calculate", price_mode="calculate")
        mapping = suggest_mapping(raw.columns)
        mapping["faturamento"] = mapping["preco_unitario"]
        with self.assertRaisesRegex(ValueError, "diferente"):
            map_sales_data(raw, mapping)

    def test_mapped_optional_column_without_observations_is_unavailable(self):
        raw = standard_frame()
        raw["categoria"] = None
        data = map_sales_data(raw, suggest_mapping(raw.columns))
        self.assertNotIn("categoria", available_dimensions(data))

    def test_empty_missing_product_and_boolean_numbers_are_rejected(self):
        raw = standard_frame().astype(object)
        raw.loc[0, "produto"] = ""
        raw.loc[1, "quantidade"] = True
        with self.assertRaises(SalesValidationError) as failure:
            prepare_sales_data(raw)
        self.assertEqual(failure.exception.report["invalid_rows"], 2)
        with self.assertRaises(SalesValidationError):
            prepare_sales_data(raw.iloc[:0])

    def test_numeric_dates_and_duplicate_schema_do_not_get_silently_reinterpreted(self):
        raw = standard_frame().astype(object)
        raw.loc[0, "data"] = 20240101
        with self.assertRaises(SalesValidationError):
            prepare_sales_data(raw)
        duplicate = pd.concat([standard_frame(), standard_frame()[["faturamento"]]], axis=1)
        with self.assertRaisesRegex(SalesValidationError, "duplicados"):
            prepare_sales_data(duplicate)


if __name__ == "__main__":
    unittest.main()
