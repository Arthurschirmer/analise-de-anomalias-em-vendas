from dataclasses import dataclass
from src.analysis.revenue_decomposition import decompose_revenue_change
from src.analysis.olist_operations import compare_olist_operations

from src.analysis.anomaly_detection import detect_period_anomalies
from src.analysis.sales_metrics import (
    compare_periods,
    compare_periods_by_dimension,
    sales_by_category,
    sales_by_period,
    sales_by_product,
    sales_by_region,
    sales_by_seller,
    total_quantity,
    total_revenue,
)


@dataclass
class ToolResult:
    name: str
    parameters: dict
    result: object

    def to_dict(self):
        result = self.result
        if hasattr(result, "to_dict"):
            result = result.to_dict(orient="records")
        return {
            "name": self.name,
            "parameters": self.parameters,
            "result": result,
        }


class SalesAnalysisTools:
    def __init__(self, data):
        self.data = data
        self.history = []

    def run(self, name, **parameters):
        tool_map = {
            "decompor_faturamento": lambda: decompose_revenue_change(
                self.data, parameters["periodo_anterior"], parameters["periodo_atual"]),
            "indicadores_operacionais": lambda: compare_olist_operations(
                self.data.attrs["olist_raw_dir"], parameters["periodo_anterior"], parameters["periodo_atual"]),
            "faturamento_total": lambda: total_revenue(self.data),
            "quantidade_vendida": lambda: total_quantity(self.data),
            "vendas_por_periodo": lambda: sales_by_period(self.data),
            "vendas_por_produto": lambda: sales_by_product(self.data),
            "vendas_por_categoria": lambda: sales_by_category(self.data),
            "vendas_por_regiao": lambda: sales_by_region(self.data),
            "vendas_por_vendedor": lambda: sales_by_seller(self.data),
            "detectar_anomalias_periodo": lambda: detect_period_anomalies(
                self.data,
                metric=parameters.get("metrica", "faturamento"),
                threshold_pct=parameters.get("limite_percentual", 30.0),
                method=parameters.get("metodo", "percentual"),
                zscore_threshold=parameters.get("limite_zscore", 2.0),
                iqr_multiplier=parameters.get("multiplicador_iqr", 1.5),
                moving_average_window=parameters.get("janela_media_movel", 3),
                min_period_value=parameters.get("valor_minimo_periodo", 0.0),
            ),
            "comparar_periodos": lambda: compare_periods(
                self.data,
                parameters["periodo_anterior"],
                parameters["periodo_atual"],
            ),
            "comparar_periodos_por_dimensao": lambda: compare_periods_by_dimension(
                self.data,
                parameters["periodo_anterior"],
                parameters["periodo_atual"],
                parameters["dimensao"],
            ),
        }

        if name not in tool_map:
            raise ValueError(f"Ferramenta desconhecida: {name}")

        result = tool_map[name]()
        tool_result = ToolResult(name=name, parameters=parameters, result=result)
        self.history.append(tool_result)
        return tool_result
