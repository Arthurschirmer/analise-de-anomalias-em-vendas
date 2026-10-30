"""Retrospective order-cohort indicators. No customer identifiers are returned."""
from pathlib import Path

import pandas as pd


def compare_olist_operations(raw_dir, previous_period, current_period):
    limitations = [
        "Coortes pelo mês da compra, incluindo todos os status; diferem da receita de itens entregues.",
        "Cancelamentos são os status finais observados no arquivo, sem instante histórico da decisão.",
        "Atraso usa apenas pedidos entregues com datas válidas de entrega e previsão.",
        "Entregas pendentes e datas ausentes ficam fora do denominador de atraso; há censura.",
        "Associações entre indicadores operacionais e faturamento não demonstram causalidade.",
    ]
    result = {"available": False, "reason": None, "monthly": [], "changes": {}, "limitations": limitations}
    path = Path(raw_dir) / "olist_orders_dataset.csv"
    if not path.is_file():
        result["reason"] = "Arquivo de pedidos não disponível nesta fonte."
        return result
    columns = ["order_id", "order_status", "order_purchase_timestamp",
               "order_estimated_delivery_date", "order_delivered_customer_date"]
    try:
        orders = pd.read_csv(path, usecols=columns)
        if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
            raise ValueError("Identificadores de pedidos ausentes ou duplicados.")
        for col in columns[2:]:
            orders[col] = pd.to_datetime(orders[col], errors="coerce", format="mixed")
        if orders["order_purchase_timestamp"].isna().any():
            raise ValueError("Há datas de compra inválidas.")
        orders["month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
        for period in [previous_period, current_period]:
            part = orders.loc[orders["month"].eq(period)]
            if part.empty:
                result["reason"] = f"Não há pedidos na coorte {period}."
                return result
            valid = (part["order_status"].eq("delivered")
                     & part["order_delivered_customer_date"].notna()
                     & part["order_estimated_delivery_date"].notna()
                     & part["order_delivered_customer_date"].ge(part["order_purchase_timestamp"])
                     & part["order_estimated_delivery_date"].ge(part["order_purchase_timestamp"]))
            # Compare dates, since estimated delivery is a calendar deadline.
            late = (part.loc[valid, "order_delivered_customer_date"].dt.normalize()
                    > part.loc[valid, "order_estimated_delivery_date"].dt.normalize())
            canceled = int(part["order_status"].eq("canceled").sum())
            result["monthly"].append({
                "periodo": period, "pedidos": len(part), "cancelados": canceled,
                "taxa_cancelamento": canceled / len(part) * 100,
                "entregas_avaliaveis": int(valid.sum()), "entregas_atrasadas": int(late.sum()),
                "taxa_atraso_observada": float(late.mean() * 100) if len(late) else None,
                "pedidos_fora_denominador_atraso": int((~valid).sum()),
            })
        before, after = result["monthly"]
        for key in ["taxa_cancelamento", "taxa_atraso_observada"]:
            result["changes"][key + "_pp"] = (
                after[key] - before[key] if before[key] is not None and after[key] is not None else None)
        result["available"] = True
    except (ValueError, OSError) as error:
        result["reason"] = str(error)
    return result

