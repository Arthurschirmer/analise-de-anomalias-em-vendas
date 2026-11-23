import copy
import unittest

import pandas as pd

from src.formatting import currency_axis_label, currency_symbol, format_currency


class CurrencyFormattingTest(unittest.TestCase):
    def test_original_brl_format_is_the_default(self):
        self.assertEqual(format_currency(1234.56), "R$ 1.234,56")
        self.assertEqual(format_currency(-12.5, {}), "R$ -12,50")
        self.assertEqual(format_currency(0), "R$ 0,00")
        self.assertEqual(currency_axis_label(), "Faturamento (R$)")

    def test_gbp_preserves_amounts_and_reads_source_metadata(self):
        metadata = {"currency": "GBP", "currency_symbol": "£"}
        frame = pd.DataFrame({"faturamento": [1234.56]})
        frame.attrs["dataset_context"] = metadata.copy()
        for context in (metadata, frame, {"dataset_context": metadata}, {"contexto_da_base": metadata}):
            with self.subTest(context=type(context).__name__):
                self.assertEqual(format_currency(1234.56, context), "£ 1.234,56")
                self.assertEqual(currency_symbol(context), "£")
                self.assertEqual(currency_axis_label(context), "Faturamento (£)")
        self.assertEqual(frame.iloc[0]["faturamento"], 1234.56)

    def test_non_finite_or_missing_amounts_have_no_fake_currency_value(self):
        for value in (None, float("nan"), float("inf"), float("-inf"), pd.NA, True, "invalid"):
            with self.subTest(value=str(value)):
                self.assertEqual(format_currency(value, {"currency": "GBP"}), "—")

    def test_symbols_are_explicit_and_metadata_is_not_modified(self):
        context = {"contexto_da_base": {"currency": "GBP", "currency_symbol": "R$"}}
        original = copy.deepcopy(context)
        self.assertEqual(currency_symbol(context), "£")
        self.assertEqual(context, original)
        self.assertEqual(currency_symbol({"currency_symbol": "£"}), "£")
        self.assertEqual(currency_symbol({"currency": "JPY"}), "JPY")
        self.assertEqual(currency_symbol({"currency_symbol": "<script>"}), "R$")


if __name__ == "__main__":
    unittest.main()
