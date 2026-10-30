import unittest

import pandas as pd

from src.agent.period_parser import extract_periods_from_text


class PeriodParserTest(unittest.TestCase):
    def test_extract_iso_months_in_chronological_order(self):
        data = pd.DataFrame({"data": pd.to_datetime(["2024-07-01", "2024-08-01"])})
        periods = extract_periods_from_text(
            "Investigue a variação de faturamento de 2024-07 para 2024-08.", data
        )
        self.assertEqual(periods, ["2024-07", "2024-08"])

    def test_extract_iso_comparison_across_year_boundary(self):
        data = pd.DataFrame({"data": pd.to_datetime(["2023-12-01", "2024-01-01"])})
        periods = extract_periods_from_text(
            "Investigue 2024-01 em comparacao com 2023-12.", data
        )
        self.assertEqual(periods, ["2023-12", "2024-01"])

    def test_extract_mixed_iso_and_named_month(self):
        data = pd.DataFrame({"data": pd.to_datetime(["2024-07-01", "2024-08-01"])})
        periods = extract_periods_from_text(
            "Compare agosto de 2024 e 2024-07.", data
        )
        self.assertEqual(periods, ["2024-07", "2024-08"])

    def test_extract_periods_with_accents(self):
        data = pd.DataFrame({"data": pd.to_datetime(["2024-02-01", "2024-03-01"])})

        periods = extract_periods_from_text(
            "O faturamento caiu em mar\u00e7o em comparacao com fevereiro.",
            data,
        )

        self.assertEqual(periods, ["2024-02", "2024-03"])

    def test_extract_periods_across_year_boundary(self):
        data = pd.DataFrame({"data": pd.to_datetime(["2017-12-01", "2018-01-01"])})

        periods = extract_periods_from_text(
            "O faturamento aumentou em janeiro em comparacao com dezembro.",
            data,
        )

        self.assertEqual(periods, ["2017-12", "2018-01"])

    def test_extract_periods_with_explicit_years(self):
        data = pd.DataFrame({"data": pd.to_datetime(["2017-12-01", "2018-01-01"])})

        periods = extract_periods_from_text(
            "O faturamento aumentou em janeiro de 2018 em comparacao com dezembro de 2017.",
            data,
        )

        self.assertEqual(periods, ["2017-12", "2018-01"])


if __name__ == "__main__":
    unittest.main()
