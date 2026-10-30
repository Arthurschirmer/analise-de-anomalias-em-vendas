"""Past-only seasonal references and explicit temporal calibration.

The limits below are empirical decision limits, not confidence or prediction
intervals. No label, current value, future value or configured LLM is used to
construct a point's historical reference.
"""

import math

import numpy as np
import pandas as pd


SERIES_COLUMNS = [
    "periodo", "data", "valor", "valor_observado", "observado", "completo",
    "motivo", "inicio_periodo", "fim_periodo", "dias_observados", "dias_esperados",
]
SCORE_COLUMNS = [
    "expected", "scale", "score", "lower", "upper", "flag", "eligible",
    "n_history", "reason", "decision_threshold",
]
MAD_SCALE = 1.4826


def _finite_number(value, name, minimum=None, strictly_positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} deve ser um número finito.") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} deve ser um número finito.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} deve ser maior ou igual a {minimum}.")
    if strictly_positive and number <= 0:
        raise ValueError(f"{name} deve ser positivo.")
    return number


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} deve ser um inteiro positivo.")
    return int(value)


def _frequency(value):
    text = str(value).upper()
    if text == "W":
        text = "W-SUN"
    if text not in {"D", "W-SUN"}:
        raise ValueError("A frequência histórica deve ser D ou W-SUN.")
    return text


def _series_attrs(frequency, metric, missing_policy):
    return {
        "frequency": frequency,
        "metric": metric,
        "seasonal_period": 7 if frequency == "D" else 1,
        "missing_policy": missing_policy,
        "limits_type": "empirical_decision_limits_not_confidence_intervals",
    }


def build_time_series(data, frequency="D", metric="faturamento", missing_policy="unknown"):
    """Aggregate observed sales on a daily or Monday-to-Sunday calendar.

    With the default unknown policy, a date absent from the transaction table
    stays NaN. A weekly bin containing an unknown date or a truncated boundary
    stays incomplete and cannot produce an alert. Its observed subtotal remains
    available in valor_observado. The zero policy is an explicit assumption that
    absent dates *inside* the observed date span mean zero sales; it never pads
    a partially observed boundary week outside that span.
    """
    frequency = _frequency(frequency)
    if metric not in {"faturamento", "quantidade"}:
        raise ValueError("A métrica histórica deve ser faturamento ou quantidade.")
    if missing_policy not in {"unknown", "zero"}:
        raise ValueError("A política de datas ausentes deve ser unknown ou zero.")
    attrs = _series_attrs(frequency, metric, missing_policy)
    if data is None or data.empty:
        frame = pd.DataFrame(columns=SERIES_COLUMNS)
        frame.attrs.update(attrs)
        return frame
    if "data" not in data or metric not in data:
        raise ValueError(f"A série precisa das colunas data e {metric}.")

    dates = pd.to_datetime(data["data"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Há datas ausentes ou inválidas na série histórica.")
    if dates.dt.tz is not None:
        # Preserve each transaction's local calendar date, not a UTC conversion.
        dates = dates.dt.tz_localize(None)
    values = pd.to_numeric(data[metric], errors="coerce")
    if not values.map(lambda value: pd.notna(value) and math.isfinite(float(value))).all():
        raise ValueError("Há valores não finitos na métrica histórica.")
    observed = pd.DataFrame({"data": dates.dt.normalize(), "valor": values}).groupby("data")["valor"].sum()
    calendar = pd.date_range(observed.index.min(), observed.index.max(), freq="D")
    daily = observed.reindex(calendar)
    was_observed = daily.notna()
    if missing_policy == "zero":
        daily = daily.fillna(0.0)

    if frequency == "D":
        result = pd.DataFrame({
            "data": calendar,
            "valor": daily.to_numpy(dtype=float),
            "valor_observado": observed.reindex(calendar).to_numpy(dtype=float),
            "observado": was_observed.to_numpy(dtype=bool),
            "completo": daily.notna().to_numpy(dtype=bool),
            "dias_observados": was_observed.to_numpy(dtype=int),
            "dias_esperados": 1,
        })
        result["inicio_periodo"] = result["data"]
        result["fim_periodo"] = result["data"]
    else:
        daily_frame = pd.DataFrame({
            "data": calendar, "valor": daily.to_numpy(),
            "observado": was_observed.to_numpy(),
        })
        daily_frame["semana"] = daily_frame["data"].dt.to_period("W-SUN")
        rows = []
        for period, part in daily_frame.groupby("semana", sort=True):
            within_span = len(part) == 7
            complete = within_span and part["valor"].notna().all()
            observed_subtotal = part.loc[part["observado"], "valor"].sum(min_count=1)
            rows.append({
                "data": period.end_time.normalize(),
                "valor": float(part["valor"].sum()) if complete else float("nan"),
                "valor_observado": float(observed_subtotal) if pd.notna(observed_subtotal) else float("nan"),
                "observado": bool(part["observado"].any()),
                "completo": bool(complete),
                "inicio_periodo": period.start_time.normalize(),
                "fim_periodo": period.end_time.normalize(),
                "dias_observados": int(part["observado"].sum()),
                "dias_esperados": 7,
            })
        result = pd.DataFrame(rows)
    result["periodo"] = result["data"].dt.strftime("%Y-%m-%d")
    result["motivo"] = np.where(
        result["completo"],
        np.where(result["observado"], "observed", "assumed_zero"),
        np.where(result["observado"], "partial_period", "missing_observation"),
    )
    result = result[SERIES_COLUMNS].reset_index(drop=True)
    result.attrs.update(attrs)
    return result


def _ordered_series(series):
    frame = pd.DataFrame(series).copy()
    attrs = dict(getattr(series, "attrs", {}))
    if "data" not in frame or "valor" not in frame:
        raise ValueError("A série histórica precisa de data e valor.")
    dates = pd.to_datetime(frame["data"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("A série histórica precisa de datas válidas e únicas.")
    frame["data"] = dates
    frame["valor"] = pd.to_numeric(frame["valor"], errors="coerce")
    if np.isinf(frame["valor"].to_numpy(dtype=float)).any():
        raise ValueError("A série histórica contém valores infinitos.")
    frame = frame.sort_values("data", kind="stable").reset_index(drop=True)
    frame.attrs.update(attrs)
    return frame


def score_historical_series(
    series, seasonal_period=None, window=8, min_history=4,
    scale_floor_pct=0.05, absolute_floor=1.0, threshold=3.0, trend="none",
):
    """Score each point against only earlier positions in the same season.

    Daily defaults compare the same weekday in previous weeks. Weekly defaults
    compare previous weeks. At most window earlier seasonal positions are used;
    unknown/incomplete positions are not substituted with zero or borrowed from
    other seasons. The reference is their median, and dispersion is the larger
    of scaled MAD, a relative reference floor and an absolute measurement floor.
    """
    if trend not in {"none", "linear"}:
        raise ValueError("A tendência deve ser none ou linear.")
    window = _positive_integer(window, "A janela")
    min_history = _positive_integer(min_history, "O histórico mínimo")
    if min_history > window:
        raise ValueError("O histórico mínimo não pode exceder a janela.")
    floor_pct = _finite_number(scale_floor_pct, "O piso relativo", minimum=0)
    floor_abs = _finite_number(absolute_floor, "O piso absoluto", strictly_positive=True)
    threshold = _finite_number(threshold, "O limiar", strictly_positive=True)
    frame = _ordered_series(series)
    seasonal_period = _positive_integer(
        seasonal_period if seasonal_period is not None else frame.attrs.get("seasonal_period", 7),
        "O período sazonal",
    )
    if not frame.empty:
        frequency = _frequency(frame.attrs.get("frequency", "D"))
        expected_dates = pd.date_range(frame["data"].iloc[0], frame["data"].iloc[-1], freq=frequency)
        if len(expected_dates) != len(frame) or not np.array_equal(expected_dates.to_numpy(), frame["data"].to_numpy()):
            raise ValueError("A série precisa de calendário regular. Use build_time_series para preservar as lacunas.")

    numbers = frame["valor"].to_numpy(dtype=float)
    complete = (
        frame["completo"].fillna(False).to_numpy(dtype=bool)
        if "completo" in frame else np.isfinite(numbers)
    )
    rows = []
    for index, observed in enumerate(numbers):
        positions = range(index - seasonal_period, max(-1, index - seasonal_period * (window + 1)), -seasonal_period)
        valid_positions = [pos for pos in positions if pos >= 0 and complete[pos] and math.isfinite(numbers[pos])]
        history = [numbers[pos] for pos in valid_positions]
        count = len(history)
        expected = float(np.median(history)) if count >= min_history else float("nan")
        spread = float("nan")
        score = float("nan")
        current_valid = complete[index] and math.isfinite(observed)
        reason = (
            "missing_observation" if not math.isfinite(observed)
            else "partial_period" if not complete[index]
            else "insufficient_history" if count < min_history
            else "ready"
        )
        if count >= min_history:
            residuals = np.asarray(history) - expected
            if trend == "linear" and count >= 3:
                # Robust pairwise slope, using past points only. Gaps retain their
                # calendar distance. Extrapolation is local, not an annual forecast.
                x = np.asarray(valid_positions, dtype=float)
                y = np.asarray(history, dtype=float)
                slope = float(np.median([
                    (y[j] - y[i]) / (x[j] - x[i])
                    for i in range(count) for j in range(i + 1, count)
                ]))
                intercept = float(np.median(y - slope * x))
                expected = intercept + slope * index
                residuals = y - (intercept + slope * x)
            raw_mad = float(np.median(np.abs(residuals - np.median(residuals))))
            spread = max(MAD_SCALE * raw_mad, abs(expected) * floor_pct, floor_abs)
            if current_valid:
                score = abs(float(observed) - expected) / spread
        eligible = current_valid and count >= min_history
        rows.append({
            "expected": expected, "scale": spread, "score": score,
            "lower": expected - threshold * spread, "upper": expected + threshold * spread,
            "flag": bool(eligible and score > threshold), "eligible": bool(eligible),
            "n_history": count, "reason": reason, "decision_threshold": threshold,
        })
    scored = pd.concat([frame, pd.DataFrame(rows, columns=SCORE_COLUMNS)], axis=1)
    scored.attrs.update(frame.attrs)
    scored.attrs.update(
        baseline="past_only_seasonal_median_mad", trend=trend,
        seasonal_period=seasonal_period, window=window, min_history=min_history,
        scale_floor_pct=floor_pct, absolute_floor=floor_abs,
        limits_type="empirical_decision_limits_not_confidence_intervals",
    )
    return scored


def temporal_split(
    series, train_fraction=0.5, calibration_fraction=0.25,
    min_train=28, min_calibration=7, min_test=7,
):
    """Split a regular chronological series into disjoint calendar sections."""
    train_fraction = _finite_number(train_fraction, "A fração de treino", strictly_positive=True)
    calibration_fraction = _finite_number(calibration_fraction, "A fração de calibração", strictly_positive=True)
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("Treino e calibração devem deixar uma fração para teste.")
    min_train = _positive_integer(min_train, "O tamanho mínimo de treino")
    min_calibration = _positive_integer(min_calibration, "O tamanho mínimo de calibração")
    min_test = _positive_integer(min_test, "O tamanho mínimo de teste")
    frame = _ordered_series(series)
    n_train = int(len(frame) * train_fraction)
    n_calibration = int(len(frame) * calibration_fraction)
    n_test = len(frame) - n_train - n_calibration
    if n_train < min_train or n_calibration < min_calibration or n_test < min_test:
        raise ValueError(
            "Histórico insuficiente para separar treino, calibração e teste: "
            f"{n_train}/{n_calibration}/{n_test} períodos disponíveis."
        )
    frame["split"] = ["train"] * n_train + ["calibration"] * n_calibration + ["test"] * n_test
    boundaries = {}
    for label, part in frame.groupby("split", sort=False):
        boundaries[label] = {
            "start": part["data"].iloc[0].date().isoformat(),
            "end": part["data"].iloc[-1].date().isoformat(),
            "count": len(part),
        }
    frame.attrs["split_boundaries"] = boundaries
    frame.attrs["split_method"] = "chronological_disjoint"
    return frame


def calibrate_threshold(
    scored, quantile=0.98, minimum=3.0, min_calibration_scores=5,
    method="quantile", robust_multiplier=3.0,
):
    """Freeze an empirical threshold from calibration scores only.

    Unlabelled real calibration values cannot establish a true false-positive
    rate. quantile is an operational reference, not an accuracy guarantee.
    """
    quantile = _finite_number(quantile, "O quantil", strictly_positive=True)
    minimum = _finite_number(minimum, "O limiar mínimo", strictly_positive=True)
    robust_multiplier = _finite_number(robust_multiplier, "O multiplicador robusto", strictly_positive=True)
    if method not in {"quantile", "robust"}:
        raise ValueError("O método de calibração deve ser quantile ou robust.")
    min_calibration_scores = _positive_integer(min_calibration_scores, "A quantidade mínima de escores")
    if quantile > 1:
        raise ValueError("O quantil deve estar entre zero e um.")
    required = {"data", "split", "score", "eligible"}
    if not required.issubset(scored.columns):
        raise ValueError("A calibração requer data, split, score e eligible.")
    dates = pd.to_datetime(scored["data"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Há datas inválidas na calibração.")
    # Validate ordering and non-overlap, even if callers supplied split labels.
    ordered = pd.DataFrame({"data": dates, "split": scored["split"]}).sort_values("data")
    labels = list(dict.fromkeys(ordered["split"].tolist()))
    if labels != ["train", "calibration", "test"] or ordered["data"].duplicated().any():
        raise ValueError("A calibração exige blocos cronológicos distintos de treino, calibração e teste.")
    ranks = ordered["split"].map({"train": 0, "calibration": 1, "test": 2})
    if not ranks.is_monotonic_increasing:
        raise ValueError("As partições temporais se sobrepõem ou estão intercaladas.")

    values = pd.to_numeric(scored["score"], errors="coerce")
    mask = scored["split"].eq("calibration") & scored["eligible"].fillna(False) & values.map(
        lambda value: pd.notna(value) and math.isfinite(float(value))
    )
    calibration = values.loc[mask]
    if len(calibration) < min_calibration_scores:
        raise ValueError("Histórico de calibração insuficiente para congelar um limiar.")
    median = float(calibration.median())
    raw_mad = float((calibration - median).abs().median())
    robust_scale = MAD_SCALE * raw_mad
    quantile_threshold = max(minimum, float(calibration.quantile(quantile, interpolation="higher")))
    robust_threshold = max(minimum, median + robust_multiplier * robust_scale)
    threshold = robust_threshold if method == "robust" else quantile_threshold
    selected_dates = dates.loc[mask]
    # Diagnostic only: this is an observed calibration exceedance share, not a
    # measured false-positive rate or a guaranteed future alert frequency.
    diagnostics = {
        "score_median": median, "score_mad": raw_mad,
        "score_scaled_mad": robust_scale, "score_max": float(calibration.max()),
        "quantile_threshold": quantile_threshold, "robust_threshold": robust_threshold,
        "robust_multiplier": robust_multiplier,
        "calibration_exceedances": int(calibration.gt(threshold).sum()),
        "calibration_exceedance_share": float(calibration.gt(threshold).mean()),
        "tail_inflation_ratio": quantile_threshold / robust_threshold,
        "tail_sensitive": bool(quantile_threshold > 2 * robust_threshold),
    }
    return {
        "threshold": threshold, "quantile": quantile, "minimum": minimum,
        "n_calibration": len(calibration),
        "calibration_start": selected_dates.min().date().isoformat(),
        "calibration_end": selected_dates.max().date().isoformat(),
        "method": "calibration_robust_median_mad" if method == "robust" else "calibration_empirical_quantile",
        "method_key": method, "diagnostics": diagnostics,
        "frozen": True,
        "limitation": (
            "Limiar operacional estimado somente na calibração; sem rótulos reais não mede taxa de falsos positivos. "
            "A regra robusta reduz a influência de uma cauda extrema, mas pode aumentar alertas em dados voláteis. "
            "Método e parâmetros devem ser declarados antes da avaliação de teste."
        ),
    }


def apply_threshold(scored, threshold):
    """Apply one frozen threshold to a scored series without fitting anything."""
    calibration = dict(threshold) if isinstance(threshold, dict) else None
    threshold = _finite_number(
        calibration["threshold"] if calibration else threshold, "O limiar", strictly_positive=True,
    )
    required = {"expected", "scale", "score", "eligible"}
    if not required.issubset(scored.columns):
        raise ValueError("A aplicação do limiar requer expected, scale, score e eligible.")
    result = scored.copy()
    result["decision_threshold"] = threshold
    result["lower"] = result["expected"] - threshold * result["scale"]
    result["upper"] = result["expected"] + threshold * result["scale"]
    result["flag"] = result["eligible"].fillna(False) & result["score"].gt(threshold)
    result.attrs.update(getattr(scored, "attrs", {}))
    if calibration:
        result.attrs["calibration"] = calibration
    return result

