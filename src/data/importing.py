"""Explicit parsing and column mapping for local CSV/Excel uploads."""

from datetime import date, datetime
from io import BytesIO
from numbers import Number
from pathlib import Path
import unicodedata
from zipfile import BadZipFile, ZipFile

import pandas as pd

from src.data.loader import STANDARD_COLUMNS, prepare_sales_data


MAX_FILE_BYTES = 64 * 1024 * 1024
OPTIONAL_DIMENSIONS = ("categoria", "regiao", "vendedor")
ALIASES = {
    "data": ("data", "date", "invoicedate", "orderdate", "datavenda"),
    "produto": ("produto", "product", "productid", "stockcode", "productname"),
    "categoria": ("categoria", "category", "productcategory", "family"),
    "regiao": ("regiao", "region", "country", "state", "pais"),
    "vendedor": ("vendedor", "seller", "sellerid", "salesperson"),
    "quantidade": ("quantidade", "quantity", "qty", "units", "unitssold"),
    "preco_unitario": ("precounitario", "unitprice", "price", "priceeach"),
    "desconto": ("desconto", "discount", "discountpercent"),
    "faturamento": ("faturamento", "revenue", "sales", "total", "totalamount", "amount"),
}


def _normalized(value):
    return "".join(char for char in unicodedata.normalize("NFKD", str(value).casefold()) if char.isalnum())


def suggest_mapping(columns):
    """Suggest, never silently apply, correspondences to the internal schema."""
    normalized = {_normalized(column): column for column in columns}
    return {field: next((normalized[name] for name in names if name in normalized), None)
            for field, names in ALIASES.items()}


def read_tabular_file(content, filename, *, separator=",", encoding="utf-8-sig", sheet=None):
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Escolha um arquivo não vazio de até 64 MB.")
    extension = Path(filename).suffix.casefold()
    if extension == ".csv":
        if separator not in {",", ";", "\t", "|"}:
            raise ValueError("Separador de CSV inválido.")
        frame = pd.read_csv(BytesIO(content), sep=separator, encoding=encoding, dtype="string")
        frame.attrs["read_options"] = {"separator": separator, "encoding": encoding}
        return frame, []
    if extension != ".xlsx":
        raise ValueError("A importação guiada aceita CSV ou Excel .xlsx.")
    try:
        with ZipFile(BytesIO(content)) as archive:
            if len(archive.infolist()) > 1000 or sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024:
                raise ValueError("A planilha excede os limites desta importação.")
        with pd.ExcelFile(BytesIO(content), engine="openpyxl") as workbook:
            names = workbook.sheet_names
            selected = sheet if sheet is not None else names[0]
            if selected not in names:
                raise ValueError("Escolha uma aba existente na planilha.")
            frame = workbook.parse(selected, dtype=object)
            frame.attrs["read_options"] = {"sheet": selected}
            return frame, names
    except BadZipFile as error:
        raise ValueError("O arquivo não é uma planilha .xlsx válida.") from error


def _numbers(values, decimal):
    def convert(value):
        if isinstance(value, Number) and not isinstance(value, bool):
            return value
        if not isinstance(value, str):
            return float("nan")
        text = value.strip().replace("\u00a0", "").replace(" ", "")
        return text.replace(".", "").replace(",", ".") if decimal == "," else text.replace(",", "")
    return pd.to_numeric(values.map(convert), errors="coerce")


def map_sales_data(raw, mapping, *, currency="BRL", date_format="%Y-%m-%d", decimal=".",
                   discount_percent=False, revenue_mode="column", price_mode="column",
                   invalid_policy="raise", filename="arquivo", source_hash=None):
    if currency not in {"BRL", "GBP", "USD", "EUR"} or decimal not in {".", ","}:
        raise ValueError("Escolha uma moeda e um formato numérico válidos.")
    if date_format not in {"%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "mixed"}:
        raise ValueError("Escolha um formato de data válido.")
    if revenue_mode not in {"column", "calculate"} or price_mode not in {"column", "calculate"}:
        raise ValueError("Definição de faturamento ou preço inválida.")
    if revenue_mode == price_mode == "calculate":
        raise ValueError("Informe uma coluna de faturamento ou de preço; os dois não podem depender um do outro.")
    required = ["data", "produto", "quantidade"]
    if revenue_mode == "column":
        required.append("faturamento")
    if price_mode == "column":
        required.append("preco_unitario")
    if any(not mapping.get(field) for field in required):
        raise ValueError("Associe as colunas obrigatórias: " + ", ".join(required) + ".")
    selected = {field: mapping.get(field) for field in STANDARD_COLUMNS}
    if revenue_mode == "calculate":
        selected["faturamento"] = None
    if price_mode == "calculate":
        selected["preco_unitario"] = None
    columns = [value for value in selected.values() if value is not None]
    if any(column not in raw.columns for column in columns):
        raise ValueError("O mapeamento inclui uma coluna que não existe no arquivo.")
    if len(columns) != len(set(columns)):
        raise ValueError("Associe uma coluna diferente a cada campo, ou escolha um cálculo derivado.")
    result = pd.DataFrame(index=raw.index)
    dates = raw[selected["data"]]
    typed_dates = dates.dropna().map(lambda value: isinstance(value, (date, datetime, pd.Timestamp))).all()
    numeric_dates = dates.dropna().map(lambda value: isinstance(value, Number)).any()
    result["data"] = pd.NaT if numeric_dates else pd.to_datetime(
        dates, errors="coerce", format="mixed" if typed_dates else date_format)
    observed_fields = {}
    for field in ("produto", *OPTIONAL_DIMENSIONS):
        values = raw[selected[field]].astype("string").str.strip() if selected[field] else pd.Series(pd.NA, index=raw.index, dtype="string")
        values = values.replace("", pd.NA)
        observed_fields[field] = bool(values.notna().any())
        result[field] = values if field == "produto" else values.fillna("Não informado")
    for field in ("quantidade", "preco_unitario", "faturamento", "desconto"):
        values = raw[selected[field]] if selected[field] else None
        if field == "desconto" and discount_percent and values is not None:
            values = values.map(lambda value: value.rstrip("% ") if isinstance(value, str) else value)
        result[field] = _numbers(values, decimal) if values is not None else 0.0
    if selected["desconto"] and discount_percent:
        result["desconto"] /= 100
    if revenue_mode == "calculate":
        result["faturamento"] = result["quantidade"] * result["preco_unitario"] * (1 - result["desconto"])
    if price_mode == "calculate":
        result["preco_unitario"] = result["faturamento"].div(result["quantidade"].where(result["quantidade"].ne(0)))
    available = [field for field in ("produto", *OPTIONAL_DIMENSIONS) if selected[field] and observed_fields[field]]
    limitations = ["Colunas associadas e definições confirmadas pelo usuário; a validação estrutural não certifica a correção comercial."]
    if revenue_mode == "calculate":
        limitations.append("Faturamento calculado pela fórmula quantidade × preço × (1 − desconto), escolhida pelo usuário.")
        if not selected["desconto"]:
            limitations.append("No faturamento calculado, desconto zero foi uma hipótese explícita da fórmula, não uma informação observada.")
    result.attrs["dataset_context"] = {
        "source_type": "mapped_upload", "source_label": filename, "currency": currency,
        "available_dimensions": available, "discount_available": bool(selected["desconto"]),
        "category_available": "categoria" in available, "region_available": "regiao" in available,
        "seller_available": "vendedor" in available, "product_available": True,
        "region_label": "País" if _normalized(selected["regiao"]) in {"country", "pais"} else "Região",
        "revenue_definition": "Faturamento informado na coluna de origem." if revenue_mode == "column" else "Faturamento calculado por quantidade × preço × (1 − desconto), por escolha do usuário.",
        "price_definition": "Valor informado na coluna de preço." if price_mode == "column" else "Valor unitário efetivo calculado por faturamento ÷ quantidade.",
        "row_definition": "Uma linha do arquivo importado; não foi feita agregação nem remoção de duplicatas.",
        "limitations": limitations,
    }
    result.attrs["mapping_audit"] = {"source_file": filename, "source_sha256": source_hash,
        "mapping": selected, "date_format": date_format, "decimal": decimal,
        "revenue_mode": revenue_mode, "price_mode": price_mode, "discount_percent": discount_percent,
        "invalid_policy": invalid_policy, "read_options": dict(raw.attrs.get("read_options", {}))}
    return prepare_sales_data(result, invalid_policy=invalid_policy)
