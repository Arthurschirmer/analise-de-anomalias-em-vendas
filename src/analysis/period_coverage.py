"""Observed calendar coverage; missing rows never establish collection failure."""

import pandas as pd


def comparison_coverage(data, previous, current):
    dates = pd.to_datetime(data["data"])
    periods = dates.dt.to_period("M").astype(str)
    result = []
    for month in (str(previous), str(current)):
        part = dates.loc[periods.eq(month)]
        if part.empty:
            raise ValueError("Os dois meses precisam ter registros na base.")
        first, last = part.min(), part.max()
        result.append({"period": month, "first_date": first.date().isoformat(), "last_date": last.date().isoformat(),
                       "first_day": int(first.day), "last_day": int(last.day),
                       "observed_days": int(part.dt.normalize().nunique()), "calendar_days": int(first.days_in_month),
                       "boundary_gap": bool(first.day > 1 or last.day < last.days_in_month)})
    return {"previous": result[0], "current": result[1],
            "needs_attention": any(item["boundary_gap"] for item in result),
            "first_common_day": max(item["first_day"] for item in result),
            "last_common_day": min(item["last_day"] for item in result)}


def select_comparison_window(data, previous, current, align_days=False):
    if not align_days:
        return data
    coverage = comparison_coverage(data, previous, current)
    start, end = coverage["first_common_day"], coverage["last_common_day"]
    if start > end:
        raise ValueError("Os meses não têm um intervalo de dias em comum para esse recorte.")
    dates = pd.to_datetime(data["data"])
    mask = dates.dt.to_period("M").astype(str).isin([str(previous), str(current)]) & dates.dt.day.between(start, end)
    result = data.loc[mask].copy()
    if result.empty or result["data"].dt.to_period("M").nunique() != 2:
        raise ValueError("O intervalo escolhido precisa conter registros nos dois meses.")
    description = f"Comparação somente dos dias {start} a {end} em {previous} e {current}. Dias sem registros não foram preenchidos. O recorte não iguala feriados ou dias da semana."
    context = dict(result.attrs.get("dataset_context", {}))
    context.update(comparison_window={"mode": "aligned_days", "first_day": start, "last_day": end,
                                      "previous": str(previous), "current": str(current)},
                   comparison_window_definition=description,
                   limitations=[*context.get("limitations", []), description])
    if result.attrs.pop("olist_raw_dir", None):
        context["limitations"].append("Indicadores operacionais mensais da Olist não são usados neste recorte de dias.")
    result.attrs["dataset_context"] = context
    return result


def report_comparison_data(data, report):
    window = report.get("contexto_da_base", {}).get("comparison_window", {})
    if window.get("mode") != "aligned_days":
        return data
    dates = pd.to_datetime(data["data"])
    mask = dates.dt.to_period("M").astype(str).isin([window["previous"], window["current"]]) & dates.dt.day.between(window["first_day"], window["last_day"])
    return data.loc[mask].copy()
