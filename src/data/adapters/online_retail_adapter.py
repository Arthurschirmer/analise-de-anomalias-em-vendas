"""Importação explícita da base Online Retail da UCI, sem inventar campos."""

from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path, PurePosixPath
import re
import stat
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.data.adapters.base_adapter import SalesDataAdapter
from src.data.loader import STANDARD_COLUMNS, validate_sales_data


class OnlineRetailAdapter(SalesDataAdapter):
    """Converte uma planilha UCI ou seu ZIP de distribuição para vendas diárias.

    O escopo são linhas com quantidade e preço positivos e fatura não cancelada.
    Devoluções não são tratadas como anomalias nem subtraídas da receita exibida.
    Quando há várias abas, a seleção deve ser explícita pelo argumento sheet_name.
    """

    REQUIRED_COLUMNS = (
        "InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "Country"
    )
    MAX_INPUT_BYTES = 128 * 1024 * 1024
    MAX_OUTER_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
    MAX_WORKBOOK_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
    MAX_ARCHIVE_ENTRIES = 1000

    def __init__(self, input_path, sheet_name=None):
        self.input_path = Path(input_path)
        self.sheet_name = sheet_name

    @staticmethod
    def _validate_archive(archive, max_uncompressed):
        entries = archive.infolist()
        if len(entries) > OnlineRetailAdapter.MAX_ARCHIVE_ENTRIES:
            raise ValueError("O arquivo ZIP contém entradas demais para esta importação.")
        total_size = 0
        names = set()
        for entry in entries:
            name = entry.filename.replace("\\", "/")
            parts = PurePosixPath(name)
            if (
                parts.is_absolute() or ".." in parts.parts
                or re.match(r"^[A-Za-z]:", name)
                or stat.S_ISLNK(entry.external_attr >> 16)
                or entry.flag_bits & 1
            ):
                raise ValueError("O ZIP contém uma entrada insegura ou protegida por senha.")
            if name in names:
                raise ValueError("O ZIP contém nomes de entrada duplicados.")
            names.add(name)
            total_size += entry.file_size
            if total_size > max_uncompressed:
                raise ValueError("O tamanho descompactado do ZIP excede o limite de importação.")
        return entries

    def _workbook_bytes(self):
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Arquivo Online Retail não encontrado: {self.input_path}")
        if self.input_path.stat().st_size > self.MAX_INPUT_BYTES:
            raise ValueError("O arquivo Online Retail excede o limite de 128 MB.")
        suffix = self.input_path.suffix.casefold()
        if suffix not in {".xlsx", ".zip"}:
            raise ValueError("Selecione a planilha .xlsx da Online Retail ou seu arquivo .zip.")
        source_bytes = self.input_path.read_bytes()
        source_hash = sha256(source_bytes).hexdigest()
        member_name = None
        workbook_bytes = source_bytes
        try:
            if suffix == ".zip":
                with ZipFile(BytesIO(source_bytes)) as archive:
                    entries = self._validate_archive(archive, self.MAX_OUTER_UNCOMPRESSED_BYTES)
                    candidates = [
                        entry for entry in entries
                        if not entry.is_dir() and entry.filename.casefold().endswith(".xlsx")
                    ]
                    if len(candidates) != 1:
                        raise ValueError("O ZIP deve conter exatamente uma planilha .xlsx da Online Retail.")
                    member_name = candidates[0].filename
                    if candidates[0].file_size > self.MAX_INPUT_BYTES:
                        raise ValueError("A planilha no ZIP excede o limite de 128 MB.")
                    workbook_bytes = archive.read(candidates[0])
            with ZipFile(BytesIO(workbook_bytes)) as workbook_archive:
                self._validate_archive(workbook_archive, self.MAX_WORKBOOK_UNCOMPRESSED_BYTES)
        except BadZipFile as exc:
            raise ValueError("O ZIP ou a planilha .xlsx está inválido ou incompleto.") from exc
        return workbook_bytes, source_hash, member_name

    def _read(self, workbook_bytes):
        try:
            with pd.ExcelFile(BytesIO(workbook_bytes), engine="openpyxl") as workbook:
                sheets = workbook.sheet_names
                if self.sheet_name is None:
                    if len(sheets) != 1:
                        raise ValueError(
                            "A planilha contém várias abas; informe sheet_name para escolher uma."
                        )
                    selected_sheet = sheets[0]
                else:
                    if self.sheet_name not in sheets:
                        raise ValueError("A aba informada não existe na planilha Online Retail.")
                    selected_sheet = self.sheet_name
                # Description e CustomerID não são lidos nem exportados.
                raw = pd.read_excel(
                    workbook,
                    sheet_name=selected_sheet,
                    usecols=lambda name: name in self.REQUIRED_COLUMNS,
                    dtype={"InvoiceNo": "string", "StockCode": "string", "Country": "string"},
                )
        except ImportError as exc:
            raise ImportError(
                "A importação de Excel precisa de openpyxl. Instale as dependências de requirements.txt."
            ) from exc
        missing = set(self.REQUIRED_COLUMNS) - set(raw.columns)
        if missing:
            raise ValueError("A planilha não possui as colunas da Online Retail: " + ", ".join(sorted(missing)))
        if raw.empty:
            raise ValueError("A planilha Online Retail não possui registros.")
        return raw, selected_sheet

    @staticmethod
    def _sum(frame, field):
        return float(frame[field].sum())

    def load(self):
        workbook_bytes, source_hash, member_name = self._workbook_bytes()
        raw, selected_sheet = self._read(workbook_bytes)
        raw["InvoiceNo"] = raw["InvoiceNo"].str.strip()
        if raw["InvoiceNo"].isna().any() or raw["InvoiceNo"].eq("").any():
            raise ValueError("Há registros sem número de fatura (InvoiceNo); confira o arquivo de origem.")
        for field in ("Quantity", "UnitPrice"):
            raw[field] = pd.to_numeric(raw[field], errors="coerce")
            finite = np.isfinite(raw[field].to_numpy(dtype=float, na_value=np.nan))
            if not finite.all():
                raise ValueError(f"Há valores ausentes ou inválidos na coluna {field}.")
        original_dates = raw["InvoiceDate"]
        if not is_datetime64_any_dtype(original_dates):
            numeric_dates = original_dates.map(
                lambda value: isinstance(value, (int, float, bool, np.number))
            )
            if numeric_dates.any():
                raise ValueError("InvoiceDate deve conter datas, não números sem formato de data.")
        dates = pd.to_datetime(original_dates, format="mixed", errors="coerce", dayfirst=True)
        if dates.isna().any():
            raise ValueError("Há datas ausentes ou inválidas na coluna InvoiceDate.")
        raw["InvoiceDate"] = dates.dt.floor("D")
        raw["line_value"] = raw["Quantity"] * raw["UnitPrice"]
        if not np.isfinite(raw["line_value"].to_numpy(dtype=float)).all():
            raise ValueError("Há valores de faturamento não finitos após multiplicar quantidade por preço.")

        remaining = pd.Series(True, index=raw.index)
        exclusions = []
        for rule, matches in (
            ("cancelled_invoice", raw["InvoiceNo"].str.upper().str.startswith("C")),
            ("non_positive_quantity", raw["Quantity"].le(0)),
            ("non_positive_unit_price", raw["UnitPrice"].le(0)),
        ):
            excluded = remaining & matches
            excluded_frame = raw.loc[excluded]
            exclusions.append({
                "rule": rule,
                "rows": int(excluded.sum()),
                "quantity": self._sum(excluded_frame, "Quantity"),
                "line_value_gbp": self._sum(excluded_frame, "line_value"),
            })
            remaining &= ~matches
        retained = raw.loc[remaining].copy()
        if retained.empty:
            raise ValueError("Nenhuma venda permanece após aplicar o escopo da Online Retail.")

        products = retained["StockCode"].str.strip().replace("", pd.NA)
        countries = retained["Country"].str.strip().replace("", pd.NA)
        standardized = pd.DataFrame({
            "data": retained["InvoiceDate"],
            "produto": products.fillna("Produto não informado"),
            "categoria": "Não informado",
            "regiao": countries.fillna("País não informado"),
            "vendedor": "Não informado",
            "quantidade": retained["Quantity"],
            "preco_unitario": retained["UnitPrice"],
            "desconto": 0.0,
            "faturamento": retained["line_value"],
        })
        grouped = standardized.groupby(
            ["data", "produto", "categoria", "regiao", "vendedor"], as_index=False, sort=False
        ).agg(quantidade=("quantidade", "sum"), faturamento=("faturamento", "sum"))
        grouped["preco_unitario"] = grouped["faturamento"] / grouped["quantidade"]
        grouped["desconto"] = 0.0
        result = grouped[STANDARD_COLUMNS].sort_values("data", kind="stable").reset_index(drop=True)
        validate_sales_data(result)

        excluded_frame = raw.loc[~remaining]
        source_quantity = self._sum(raw, "Quantity")
        source_value = self._sum(raw, "line_value")
        retained_quantity = float(result["quantidade"].sum())
        retained_value = float(result["faturamento"].sum())
        quantity_residual = source_quantity - retained_quantity - self._sum(excluded_frame, "Quantity")
        value_residual = source_value - retained_value - self._sum(excluded_frame, "line_value")
        audit = {
            "version": 1,
            "source_file": self.input_path.name,
            "source_sha256": source_hash,
            "workbook_sha256": sha256(workbook_bytes).hexdigest(),
            "archive_member": member_name,
            "sheet": selected_sheet,
            "currency": "GBP",
            "input_rows": len(raw),
            "retained_source_rows": len(retained),
            "excluded_rows": len(excluded_frame),
            "standardized_rows": len(result),
            "exclusions_ordered": exclusions,
            "repeated_analysis_rows_retained": int(retained[list(self.REQUIRED_COLUMNS)].duplicated().sum()),
            "missing_product_rows": int(products.isna().sum()),
            "missing_country_rows": int(countries.isna().sum()),
            "source_quantity_signed": source_quantity,
            "source_line_value_signed_gbp": source_value,
            "retained_quantity": retained_quantity,
            "retained_line_value_gbp": retained_value,
            "excluded_quantity_signed": self._sum(excluded_frame, "Quantity"),
            "excluded_line_value_signed_gbp": self._sum(excluded_frame, "line_value"),
            "reconciliation": {
                "rows_match": len(raw) == len(retained) + len(excluded_frame),
                "quantity_residual": quantity_residual,
                "line_value_residual_gbp": value_residual,
                "quantity_matches": math.isclose(quantity_residual, 0, abs_tol=1e-6),
                "line_value_matches": math.isclose(value_residual, 0, abs_tol=1e-6),
            },
        }
        if not all(audit["reconciliation"][name] for name in ("rows_match", "quantity_matches", "line_value_matches")):
            raise ValueError("A reconciliação da importação Online Retail não fechou; confira o arquivo.")
        result.attrs["import_audit"] = audit
        result.attrs["dataset_context"] = {
            "source_type": "online_retail",
            "source_label": "Base pública · UCI Online Retail",
            "source_url": "https://archive.ics.uci.edu/dataset/352/online+retail",
            "currency": "GBP",
            "currency_symbol": "£",
            "region_label": "País do cliente",
            "product_label": "Código do produto",
            "seller_label": "Vendedor não disponível",
            "category_available": False,
            "seller_available": False,
            "discount_available": False,
            "available_dimensions": ["produto", "regiao"],
            "revenue_definition": (
                "Soma de Quantity × UnitPrice em libras (GBP), por data da fatura, "
                "somente com quantidade e preço positivos e sem faturas iniciadas por C."
            ),
            "quantity_definition": "Unidades nas linhas com quantidade e preço positivos e sem faturas canceladas.",
            "row_definition": "Linhas de venda agregadas por dia, código do produto e país; linhas não são pedidos.",
            "price_definition": "Valor unitário efetivo em GBP: faturamento dividido pela quantidade do grupo.",
            "limitations": [
                "A moeda original é libra esterlina (GBP); não foi feita conversão para reais.",
                "Categoria, vendedor e desconto não são informados. Não informado e desconto zero são preenchimentos.",
                "Cancelamentos, quantidades não positivas e preços não positivos ficam fora do escopo de vendas; não foram classificados como anomalias.",
                "O total não é lucro nem receita líquida de devoluções, impostos ou frete.",
                "Country indica o país do cliente; StockCode é o código do produto.",
                "CustomerID e Description não são importados. Ausência de identificação do cliente não exclui uma venda.",
                "Linhas com campos de análise repetidos foram mantidas; a importação não remove duplicatas automaticamente.",
            ],
        }
        return result
