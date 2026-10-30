from pathlib import Path

import pandas as pd

from src.data.adapters.base_adapter import SalesDataAdapter
from src.data.loader import STANDARD_COLUMNS, validate_sales_data


class OlistAdapter(SalesDataAdapter):
    REQUIRED_FILES = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
    }
    TRANSLATION_FILE = "product_category_name_translation.csv"

    def __init__(self, raw_dir, order_statuses=None):
        self.raw_dir = Path(raw_dir)
        self.order_statuses = order_statuses or ["delivered"]

    def load(self):
        self._validate_files()

        orders = pd.read_csv(
            self.raw_dir / self.REQUIRED_FILES["orders"],
            usecols=[
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
            ],
        )
        items = pd.read_csv(
            self.raw_dir / self.REQUIRED_FILES["items"],
            usecols=["order_id", "product_id", "seller_id", "price"],
        )
        products = pd.read_csv(
            self.raw_dir / self.REQUIRED_FILES["products"],
            usecols=["product_id", "product_category_name"],
        )
        customers = pd.read_csv(
            self.raw_dir / self.REQUIRED_FILES["customers"],
            usecols=["customer_id", "customer_state"],
        )

        orders = orders.loc[orders["order_status"].isin(self.order_statuses)].copy()
        data = (
            items.merge(orders, on="order_id", how="inner", validate="many_to_one")
            .merge(products, on="product_id", how="left", validate="many_to_one")
            .merge(customers, on="customer_id", how="left", validate="many_to_one")
        )

        data = self._add_translated_categories(data)
        standardized = self._standardize(data)
        validate_sales_data(standardized)
        standardized["data"] = pd.to_datetime(standardized["data"], errors="raise")
        result = standardized[STANDARD_COLUMNS]
        result.attrs["dataset_context"] = {
            "source_type": "olist",
            "source_label": "Base pública · Olist",
            "region_label": "UF do cliente",
            "product_label": "ID do produto",
            "seller_label": "ID do vendedor",
            "discount_available": False,
            "revenue_definition": "Soma do preço dos itens dos pedidos selecionados, sem frete, pela data da compra.",
            "order_statuses": list(self.order_statuses),
            "row_definition": "Itens agregados por dia/produto/categoria/UF/vendedor; linhas não equivalem a pedidos.",
            "price_definition": "Preço médio por linha agregada; mudanças no mix de produtos podem alterar a média.",
            "limitations": [
                "A Olist não fornece desconto neste adaptador. O zero é um preenchimento, não uma observação de desconto.",
                "O campo região corresponde à UF do cliente; produto e vendedor são identificadores anônimos.",
                "O valor dos itens exclui frete e não permite calcular lucro ou receita contábil.",
                "Variações de preço médio podem refletir composição das vendas e não uma mudança no preço do mesmo produto.",
            ],
        }
        result.attrs["olist_raw_dir"] = str(self.raw_dir.resolve())
        return result

    def _validate_files(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Pasta da base Olist nao encontrada: {self.raw_dir}")

        missing = [
            filename
            for filename in self.REQUIRED_FILES.values()
            if not (self.raw_dir / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Arquivos obrigatorios da Olist nao encontrados: "
                + ", ".join(sorted(missing))
            )

    def _add_translated_categories(self, data):
        translation_path = self.raw_dir / self.TRANSLATION_FILE
        if not translation_path.exists():
            data["categoria_padronizada"] = data["product_category_name"]
            return data

        translation = pd.read_csv(translation_path)
        data = data.merge(translation, on="product_category_name", how="left", validate="many_to_one")
        data["categoria_padronizada"] = data[
            "product_category_name_english"
        ].combine_first(data["product_category_name"])
        return data

    def _standardize(self, data):
        standardized = pd.DataFrame(
            {
                "data": pd.to_datetime(
                    data["order_purchase_timestamp"], errors="raise"
                ).dt.floor("D"),
                "produto": data["product_id"].fillna("sem_produto"),
                "categoria": data["categoria_padronizada"].fillna("sem_categoria"),
                "regiao": data["customer_state"].fillna("sem_regiao"),
                "vendedor": data["seller_id"].fillna("sem_vendedor"),
                "quantidade": 1,
                "preco_unitario": pd.to_numeric(data["price"], errors="raise"),
                "desconto": 0.0,
                "faturamento": pd.to_numeric(data["price"], errors="raise"),
            }
        )

        grouped = (
            standardized.groupby(
                ["data", "produto", "categoria", "regiao", "vendedor"],
                as_index=False,
            )
            .agg(
                quantidade=("quantidade", "sum"),
                faturamento=("faturamento", "sum"),
                desconto=("desconto", "mean"),
            )
            .sort_values("data")
        )
        grouped["preco_unitario"] = grouped["faturamento"] / grouped["quantidade"]
        return grouped[STANDARD_COLUMNS]
