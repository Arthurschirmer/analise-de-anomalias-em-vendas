import pandas as pd

from src.analysis.sales_metrics import percentage_change, sales_by_period


MONTH_NAMES_PT = {
    "01": "janeiro",
    "02": "fevereiro",
    "03": "marco",
    "04": "abril",
    "05": "maio",
    "06": "junho",
    "07": "julho",
    "08": "agosto",
    "09": "setembro",
    "10": "outubro",
    "11": "novembro",
    "12": "dezembro",
}


def detect_period_anomalies(
    data,
    metric="faturamento",
    freq="M",
    threshold_pct=30.0,
    method="percentual",
    zscore_threshold=2.0,
    iqr_multiplier=1.5,
    moving_average_window=3,
    min_period_value=0.0,
):
    period_sales = sales_by_period(data, freq=freq).sort_values("periodo").reset_index(
        drop=True
    )

    if metric not in period_sales.columns:
        raise ValueError(f"Metrica indisponivel para deteccao: {metric}")
    if method not in {"percentual", "zscore", "iqr", "media_movel", "combinado"}:
        raise ValueError(f"Metodo de deteccao indisponivel: {method}")

    rows = []
    history_start = 0
    for index in range(1, len(period_sales)):
        previous_row = period_sales.iloc[index - 1]
        current_row = period_sales.iloc[index]
        previous_period = pd.Period(previous_row["periodo"], freq=freq)
        current_period = pd.Period(current_row["periodo"], freq=freq)
        if current_period != previous_period + 1:
            # A missing calendar period is unknown, not a zero-sales period.
            # Restart statistical history after the gap as well.
            history_start = index
            continue
        previous_values = [
            float(value) for value in period_sales.loc[history_start : index - 1, metric].tolist()
        ]
        previous_value = float(previous_row[metric])
        current_value = float(current_row[metric])
        variation = current_value - previous_value
        variation_pct = percentage_change(previous_value, current_value)

        percentage_result = _check_percentage(variation_pct, threshold_pct)
        zscore_result = _check_zscore(
            previous_values,
            current_value,
            zscore_threshold,
        )
        iqr_result = _check_iqr(previous_values, current_value, iqr_multiplier)
        moving_average_result = _check_moving_average(
            previous_values,
            current_value,
            moving_average_window,
            threshold_pct,
        )
        has_minimum_values = (
            previous_value >= float(min_period_value)
            and current_value >= float(min_period_value)
        )
        method_results = {
            "percentual": percentage_result,
            "zscore": zscore_result,
            "iqr": iqr_result,
            "media_movel": moving_average_result,
        }
        methods_to_check = (
            list(method_results.keys()) if method == "combinado" else [method]
        )
        triggered_methods = [
            name for name in methods_to_check if method_results[name]["anomalia"]
        ]
        if not has_minimum_values:
            triggered_methods = []
        severity_score = _calculate_severity_score(
            percentage_result,
            zscore_result,
            moving_average_result,
        )

        rows.append(
            {
                "periodo_anterior": previous_row["periodo"],
                "periodo_atual": current_row["periodo"],
                "metrica": metric,
                "valor_anterior": previous_value,
                "valor_atual": current_value,
                "variacao": variation,
                "variacao_percentual": variation_pct,
                "direcao": "queda" if variation < 0 else "aumento",
                "anomalia": bool(triggered_methods),
                "metodo": method,
                "metodos_disparados": triggered_methods,
                "limite_percentual": float(threshold_pct),
                "valor_minimo_periodo": float(min_period_value),
                "base_suficiente": has_minimum_values,
                "z_score": zscore_result["z_score"],
                "limite_zscore": float(zscore_threshold),
                "limite_inferior_iqr": iqr_result["limite_inferior"],
                "limite_superior_iqr": iqr_result["limite_superior"],
                "multiplicador_iqr": float(iqr_multiplier),
                "media_movel": moving_average_result["media_movel"],
                "janela_media_movel": int(moving_average_window),
                "variacao_percentual_media_movel": moving_average_result[
                    "variacao_percentual_media_movel"
                ],
                "pontuacao_anomalia": severity_score,
            }
        )

    return rows


def get_most_relevant_anomaly(anomalies):
    detected = [item for item in anomalies if item["anomalia"]]
    if not detected:
        return None
    return sorted(
        detected,
        key=lambda item: item.get("pontuacao_anomalia")
        or abs(item["variacao_percentual"] or 0),
        reverse=True,
    )[0]


def build_anomaly_description(anomaly):
    previous_period = _period_to_text(anomaly["periodo_anterior"])
    current_period = _period_to_text(anomaly["periodo_atual"])
    metric = anomaly["metrica"]
    direction = "caiu" if anomaly["direcao"] == "queda" else "aumentou"
    return (
        f"O {metric} {direction} em {current_period} "
        f"em comparacao com {previous_period}."
    )


def _period_to_text(period):
    parts = period.split("-")
    if len(parts) == 2 and parts[1] in MONTH_NAMES_PT:
        return f"{MONTH_NAMES_PT[parts[1]]} de {parts[0]}"
    return period


def _check_percentage(variation_pct, threshold_pct):
    is_anomaly = (
        variation_pct is not None and abs(variation_pct) >= float(threshold_pct)
    )
    return {
        "anomalia": is_anomaly,
        "variacao_percentual": variation_pct,
    }


def _check_zscore(previous_values, current_value, zscore_threshold):
    if len(previous_values) < 2:
        return {"anomalia": False, "z_score": None}

    mean_value = sum(previous_values) / len(previous_values)
    variance = sum((value - mean_value) ** 2 for value in previous_values) / len(
        previous_values
    )
    std_dev = variance**0.5

    if std_dev == 0:
        z_score = 0.0 if current_value == mean_value else float("inf")
    else:
        z_score = (current_value - mean_value) / std_dev

    return {
        "anomalia": abs(z_score) >= float(zscore_threshold),
        "z_score": z_score,
    }


def _check_iqr(previous_values, current_value, iqr_multiplier):
    if len(previous_values) < 4:
        return {
            "anomalia": False,
            "limite_inferior": None,
            "limite_superior": None,
        }

    sorted_values = sorted(previous_values)
    q1 = _percentile(sorted_values, 25)
    q3 = _percentile(sorted_values, 75)
    iqr = q3 - q1
    lower_limit = q1 - float(iqr_multiplier) * iqr
    upper_limit = q3 + float(iqr_multiplier) * iqr

    return {
        "anomalia": current_value < lower_limit or current_value > upper_limit,
        "limite_inferior": lower_limit,
        "limite_superior": upper_limit,
    }


def _check_moving_average(
    previous_values,
    current_value,
    moving_average_window,
    threshold_pct,
):
    window = int(moving_average_window)
    if len(previous_values) < window or window <= 0:
        return {
            "anomalia": False,
            "media_movel": None,
            "variacao_percentual_media_movel": None,
        }

    baseline_values = previous_values[-window:]
    moving_average = sum(baseline_values) / len(baseline_values)
    variation_pct = percentage_change(moving_average, current_value)
    is_anomaly = (
        variation_pct is not None and abs(variation_pct) >= float(threshold_pct)
    )

    return {
        "anomalia": is_anomaly,
        "media_movel": moving_average,
        "variacao_percentual_media_movel": variation_pct,
    }


def _calculate_severity_score(
    percentage_result,
    zscore_result,
    moving_average_result,
):
    scores = []
    if percentage_result["variacao_percentual"] is not None:
        scores.append(abs(percentage_result["variacao_percentual"]))
    if moving_average_result["variacao_percentual_media_movel"] is not None:
        scores.append(abs(moving_average_result["variacao_percentual_media_movel"]))
    if zscore_result["z_score"] is not None:
        z_score = zscore_result["z_score"]
        scores.append(999.0 if z_score == float("inf") else abs(z_score) * 10)
    return max(scores) if scores else 0.0


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return None

    position = (len(sorted_values) - 1) * (percentile / 100)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction
