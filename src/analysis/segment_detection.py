"""Past-only anomaly detection for categories or customer states.

Selection uses training support only. Each selected segment uses the same global
calendar and temporal blocks, with its own past reference and frozen calibration.
Groups skipped for sparse data are explicitly reported, never classified normal.
"""

import math

import numpy as np
import pandas as pd

from src.analysis.historical_detection import (
    SERIES_COLUMNS, apply_threshold, build_time_series, calibrate_threshold,
    score_historical_series, temporal_split,
)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} deve ser um inteiro maior ou igual a {minimum}.")
    return int(value)


def _number(value, name, minimum=0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} deve ser um número finito.") from None
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} deve ser um número finito maior ou igual a {minimum}.")
    return result


def _configuration(configuration):
    result = {
        "frequency": "D", "metric": "faturamento", "window": 8, "min_history": 4,
        "trend": "linear", "quantile": .98, "missing_policy": "unknown",
        "calibration_method": "robust", "robust_multiplier": 3.0,
        "minimum_threshold": 3.0, "scale_floor_pct": .05, "absolute_floor": 1.0,
    }
    result.update(configuration or {})
    if result["frequency"] == "W":
        result["frequency"] = "W-SUN"
    if result["frequency"] not in {"D", "W-SUN"}:
        raise ValueError("A frequência deve ser D ou W-SUN.")
    if result["missing_policy"] not in {"unknown", "zero"}:
        raise ValueError("A política de datas ausentes deve ser unknown ou zero.")
    if result["metric"] not in {"faturamento", "quantidade"}:
        raise ValueError("A métrica deve ser faturamento ou quantidade.")
    if result["calibration_method"] not in {"robust", "quantile"}:
        raise ValueError("O método de calibração deve ser robust ou quantile.")
    return result


def _aligned_segment_series(part, calendar, global_days, config):
    """Keep every global period, including segment gaps and truncated weeks.

    Explicit zero is allowed on globally observed days after the segment first
    appears. Before its first appearance and in global gaps, activity is unknown.
    This rule never needs the segment's future last appearance.
    """
    dates = pd.to_datetime(part["data"]).dt.normalize()
    values = pd.to_numeric(part[config["metric"]], errors="raise")
    observed = pd.DataFrame({"data": dates, "valor": values}).groupby("data")["valor"].sum()
    full_days = pd.date_range(global_days.index.min(), global_days.index.max(), freq="D")
    daily = observed.reindex(full_days)
    was_observed = daily.notna()
    if config["missing_policy"] == "zero":
        can_assume_zero = global_days.reindex(full_days, fill_value=False) & (full_days >= observed.index.min())
        daily = daily.mask(daily.isna() & can_assume_zero, 0.0)
    frame = calendar[SERIES_COLUMNS].copy()
    if config["frequency"] == "D":
        frame["valor"] = daily.to_numpy(dtype=float)
        frame["valor_observado"] = observed.reindex(full_days).to_numpy(dtype=float)
        frame["observado"] = was_observed.to_numpy(dtype=bool)
        frame["completo"] = daily.notna().to_numpy(dtype=bool)
        frame["dias_observados"] = was_observed.to_numpy(dtype=int)
    else:
        day_table = pd.DataFrame({"data": full_days, "valor": daily.to_numpy(),
                                  "observado": was_observed.to_numpy()})
        day_table["periodo"] = day_table["data"].dt.to_period("W-SUN")
        grouped = {str(period): rows for period, rows in day_table.groupby("periodo", sort=True)}
        for index, row in frame.iterrows():
            period = str(pd.Timestamp(row["data"]).to_period("W-SUN"))
            rows = grouped[period]
            complete = len(rows) == 7 and rows["valor"].notna().all()
            subtotal = rows.loc[rows["observado"], "valor"].sum(min_count=1)
            frame.at[index, "valor"] = float(rows["valor"].sum()) if complete else np.nan
            frame.at[index, "valor_observado"] = subtotal
            frame.at[index, "observado"] = bool(rows["observado"].any())
            frame.at[index, "completo"] = bool(complete)
            frame.at[index, "dias_observados"] = int(rows["observado"].sum())
    frame["motivo"] = np.where(
        frame["completo"], np.where(frame["observado"], "observed_or_explicit_zero", "assumed_zero"),
        np.where(frame["observado"], "partial_period", "missing_observation"))
    frame["split"] = calendar["split"]
    frame.attrs.update(calendar.attrs)
    frame.attrs["missing_policy"] = config["missing_policy"]
    return frame


def detect_segment_anomalies(
    data, dimension="categoria", configuration=None, max_segments=12,
    min_train_observations=28, min_train_share=.01, min_expected_value=0,
):
    """Detect categories or states chosen solely by training support.

    configuration accepts the historical UI parameters and calibration_method
    (robust by default), robust_multiplier, minimum_threshold, scale_floor_pct,
    absolute_floor. The global calendar defines all split boundaries. Training
    observation counts refer to actual observed periods, excluding assumed zero.
    """
    if dimension not in {"categoria", "regiao"}:
        raise ValueError("A dimensão segmentada deve ser categoria ou regiao.")
    if data is None or data.empty or dimension not in data:
        raise ValueError("A detecção segmentada requer dados e a dimensão escolhida.")
    max_segments = _integer(max_segments, "O limite de segmentos")
    min_train_observations = _integer(min_train_observations, "As observações mínimas de treino")
    min_train_share = _number(min_train_share, "A participação mínima de treino")
    min_expected_value = _number(min_expected_value, "O valor esperado mínimo")
    if min_train_share > 1:
        raise ValueError("A participação mínima deve estar entre zero e um.")
    config = _configuration(configuration)

    # Global gaps always remain unknown here. Explicit segment zero assumes only
    # no activity for that segment on a day on which another segment is observed.
    global_series = build_time_series(
        data, frequency=config["frequency"], metric=config["metric"], missing_policy="unknown")
    weekly = config["frequency"] != "D"
    calendar = temporal_split(global_series, min_train=8 if weekly else 28,
                              min_calibration=5 if weekly else 7, min_test=5 if weekly else 7)
    daily_global = build_time_series(data, metric=config["metric"], missing_policy="unknown")
    global_days = daily_global.set_index("data")["observado"]
    dates = pd.to_datetime(data["data"], errors="raise")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    labels = data[dimension].astype("string").fillna("(não informado)").astype(str)
    frame = data.copy()
    frame["data"] = dates.dt.normalize()
    frame[config["metric"]] = pd.to_numeric(frame[config["metric"]], errors="raise")
    frame["_segmento"] = labels
    train_end = calendar.loc[calendar["split"].eq("train"), "fim_periodo"].max()
    train = frame.loc[frame["data"].le(train_end)]
    train_values = train.groupby("_segmento")[config["metric"]].sum()
    train_total = float(train_values.sum())
    # Deterministic tie breaking avoids dependence on input row order.
    ranking = sorted(train_values.index, key=lambda value: (-float(train_values[value]), value))
    ranks = {segment: rank for rank, segment in enumerate(ranking, 1)}
    rows, points = [], []
    for segment, part in frame.groupby("_segmento", sort=True):
        series = _aligned_segment_series(part, calendar, global_days, config)
        train_mask = series["split"].eq("train")
        actual_train = int((train_mask & series["observado"] & series["completo"]).sum())
        value = float(train_values.get(segment, 0.0))
        share = value / train_total if train_total > 0 else 0.0
        row = {
            "dimensao": dimension, "segmento": segment, "train_value": value,
            "train_share": share, "train_observations": actual_train,
            "train_periods": int(train_mask.sum()), "selected": False, "status": "skipped",
            "reason": "", "threshold": None, "calibration": None,
            "eligible_test_points": 0, "alerts": 0,
            "observed_test_points": int((series["split"].eq("test") & series["observado"]).sum()),
            "unknown_test_points": int((series["split"].eq("test") & ~series["completo"]).sum()),
        }
        if segment not in ranks:
            row["reason"] = "not_observed_in_training"
        elif actual_train < min_train_observations:
            row["reason"] = "insufficient_training_observations"
        elif train_total <= 0 or share < min_train_share:
            row["reason"] = "below_training_share"
        rows.append((row, series))
    support_by_segment = {row["segmento"]: not row["reason"] for row, _ in rows}
    eligible_ranking = [segment for segment in ranking if support_by_segment[segment]][:max_segments]
    for row, series in rows:
        segment = row["segmento"]
        if segment not in eligible_ranking:
            if not row["reason"]:
                row["reason"] = "outside_training_top_limit"
            continue
        row["selected"] = True
        scored = score_historical_series(
            series, window=config["window"], min_history=config["min_history"],
            trend=config["trend"], scale_floor_pct=config["scale_floor_pct"],
            absolute_floor=config["absolute_floor"])
        # A floor on expected support is declared before fitting. It filters
        # calibration and test points using their past references only.
        enough_expected = scored["expected"].ge(min_expected_value)
        scored.loc[scored["eligible"] & ~enough_expected, "reason"] = "below_expected_support"
        scored["eligible"] &= enough_expected
        scored["flag"] &= scored["eligible"]
        try:
            calibration = calibrate_threshold(
                scored, quantile=config["quantile"], method=config["calibration_method"],
                robust_multiplier=config["robust_multiplier"], minimum=config["minimum_threshold"])
        except ValueError as error:
            row["reason"] = "insufficient_calibration_support"
            row["detail"] = str(error)
            continue
        scored = apply_threshold(scored, calibration)
        scored["segmento"], scored["dimensao"] = segment, dimension
        test_mask = scored["split"].eq("test") & scored["eligible"]
        row.update(status="evaluated", reason="", threshold=calibration["threshold"],
                   calibration=calibration, eligible_test_points=int(test_mask.sum()),
                   alerts=int((test_mask & scored["flag"]).sum()))
        if not test_mask.any():
            row["reason"] = "no_eligible_test_points"
        points.append(scored)
    summary = pd.DataFrame([row for row, _ in rows])
    result_points = pd.concat(points, ignore_index=True) if points else pd.DataFrame(
        columns=[*SERIES_COLUMNS, "split", "expected", "scale", "score", "lower", "upper",
                 "flag", "eligible", "n_history", "reason", "decision_threshold", "segmento", "dimensao"])
    if result_points.empty:
        result_points = result_points.astype({"flag": bool, "eligible": bool})
    metadata = {
        "dimension": dimension, "selected_segments": eligible_ranking,
        "total_segments": len(summary), "evaluated_segments": int(summary["status"].eq("evaluated").sum()),
        "split_boundaries": calendar.attrs["split_boundaries"],
        "calibrations": {row["segmento"]: row["calibration"] for row, _ in rows if row["calibration"] is not None},
        "selection_policy": "training_support_only",
        "limits_type": "empirical_decision_limits_not_confidence_intervals",
        "limitations": [
            "Grupos escolhidos pelo suporte no treino; segmentos novos ou esparsos ficam fora da avaliação.",
            "Sem dados não significa comportamento normal; períodos inelegíveis não recebem decisão.",
            "Zero explícito só em dias globalmente observados, após a primeira observação do segmento.",
            "Muitos testes por segmento podem aumentar o número de alertas; não há controle formal de múltiplas comparações.",
            "Seleção de parâmetros é exploratória; o teste já examinado não vira uma nova validação independente.",
        ],
    }
    return {
        "points": result_points, "summary": summary,
        "configuration": {**config, "dimension": dimension, "max_segments": max_segments,
                          "min_train_observations": min_train_observations,
                          "min_train_share": min_train_share, "min_expected_value": min_expected_value},
        "metadata": metadata,
    }

