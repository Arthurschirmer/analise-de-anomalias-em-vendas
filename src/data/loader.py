from pathlib import Path

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype


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


class SalesValidationError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _inspect_sales_data(data):
    prepared = data.copy()
    report = {"input_rows": len(data), "valid_rows": 0, "invalid_rows": len(data),
              "loaded_rows": 0, "missing_columns": sorted(REQUIRED_COLUMNS - set(data.columns)),
              "duplicate_columns": data.columns[data.columns.duplicated()].astype(str).tolist(),
              "issues": [], "warnings": [], "status": "invalid"}
    invalid = pd.Series(False, index=prepared.index)
    if report["missing_columns"] or report["duplicate_columns"] or prepared.empty:
        return prepared, report, invalid

    def issue(column, mask, reason):
        nonlocal invalid
        invalid |= mask
        if mask.any():
            positions = np.flatnonzero(mask.to_numpy()) + 2
            report["issues"].append({"column": column, "reason": reason, "count": int(mask.sum()),
                                     "source_rows_sample": positions[:20].tolist()})

    try:
        if is_datetime64_any_dtype(prepared["data"]):
            dates = prepared["data"]
        else:
            numeric_dates = prepared["data"].map(lambda value: isinstance(value, (int, float, bool, np.number)))
            dates = pd.to_datetime(prepared["data"].where(~numeric_dates, pd.NA), format="mixed", errors="coerce")
        if not is_datetime64_any_dtype(dates):
            raise ValueError("mixed timezones")
    except (TypeError, ValueError):
        dates = pd.Series(pd.NaT, index=prepared.index)
    issue("data", dates.isna(), "Data ausente, inválida ou com fusos incompatíveis.")
    prepared["data"] = dates
    products = prepared["produto"].astype("string").str.strip()
    issue("produto", products.isna() | products.eq("").fillna(False), "Produto ausente; associe um código ou nome de produto.")
    for column in ("quantidade", "preco_unitario", "desconto", "faturamento"):
        values = pd.to_numeric(prepared[column], errors="coerce")
        finite = pd.Series(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)), index=prepared.index)
        booleans = prepared[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        issue(column, ~finite | booleans, "Número ausente, inválido ou não finito.")
        prepared[column] = values
    report["invalid_rows"] = int(invalid.sum())
    report["valid_rows"] = len(data) - report["invalid_rows"]
    for column in ("categoria", "regiao", "vendedor"):
        values = prepared[column].astype("string").str.strip()
        count = int((values.isna() | values.isin(["", "Não informado", "sem_categoria", "sem_vendedor", "sem_regiao"])).sum())
        if count:
            report["warnings"].append({"column": column, "count": count,
                "message": f"{column}: informação ausente ou preenchimento técnico; confira a disponibilidade dessa dimensão."})
    for column, mask, message in (
        ("quantidade", prepared["quantidade"].lt(0), "Quantidades negativas exigem uma definição para devoluções ou ajustes."),
        ("preco_unitario", prepared["preco_unitario"].le(0), "Preços não positivos exigem conferência conforme a origem."),
        ("faturamento", prepared["faturamento"].lt(0), "Valores negativos podem representar devoluções ou ajustes; não foram removidos automaticamente."),
        ("desconto", ~prepared["desconto"].between(0, 1), "O desconto esperado é uma fração; confira valores fora de 0 a 1."),
    ):
        count = int((mask & ~invalid).sum())
        if count:
            report["warnings"].append({"column": column, "count": count, "message": message})
    return prepared, report, invalid


def prepare_sales_data(data, invalid_policy="raise"):
    """Apply the same structural validation in uploads, adapters and the CLI.

    Invalid rows are excluded only with the caller's explicit ``drop`` policy.
    Negative business values remain observations with warnings, not anomalies.
    """
    if invalid_policy not in {"raise", "drop"}:
        raise ValueError("A política de linhas inválidas deve ser raise ou drop.")
    prepared, report, invalid = _inspect_sales_data(data)
    if report["missing_columns"]:
        raise SalesValidationError("O CSV nao possui as colunas obrigatorias: " + ", ".join(report["missing_columns"]), report)
    if report["duplicate_columns"]:
        raise SalesValidationError("Há nomes de colunas duplicados: " + ", ".join(report["duplicate_columns"]), report)
    if prepared.empty:
        raise SalesValidationError("A base não contém registros de vendas.", report)
    if report["invalid_rows"] and invalid_policy == "raise":
        raise SalesValidationError(f"A base contém {report['invalid_rows']} linhas com datas ou números inválidos. Confira o resumo da validação.", report)
    if not report["valid_rows"]:
        raise SalesValidationError("Nenhuma linha válida permanece para análise.", report)
    result = prepared.loc[~invalid, STANDARD_COLUMNS].copy()
    report.update(loaded_rows=len(result), status="filtered" if report["invalid_rows"] else "valid")
    result.attrs["validation_report"] = report
    result.attrs.setdefault("import_validation_report", report)
    return result


def validate_sales_data(data):
    """Validate and normalize the supplied frame, preserving the adapter API."""
    prepared = prepare_sales_data(data)
    for column in ("data", "quantidade", "preco_unitario", "desconto", "faturamento"):
        data[column] = prepared[column]
    data.attrs.update(prepared.attrs)
    return prepared.attrs["validation_report"]
