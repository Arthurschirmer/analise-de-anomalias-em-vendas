from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "data",
    "produto",
    "categoria",
    "regiao",
    "vendedor",
    "quantidade",
    "preco_unitario",
    "desconto",
    "faturamento",
}

STANDARD_COLUMNS = [
    "data",
    "produto",
    "categoria",
    "regiao",
    "vendedor",
    "quantidade",
    "preco_unitario",
    "desconto",
    "faturamento",
]


def load_sales_csv(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo CSV nao encontrado: {file_path}")

    data = pd.read_csv(file_path)
    return prepare_sales_data(data)


def prepare_sales_data(data):
    data = data.copy()
    validate_sales_data(data)
    data["data"] = pd.to_datetime(data["data"], errors="raise")
    return data[STANDARD_COLUMNS]


def validate_sales_data(data):
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"O CSV nao possui as colunas obrigatorias: {missing}")

    numeric_columns = ["quantidade", "preco_unitario", "desconto", "faturamento"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
