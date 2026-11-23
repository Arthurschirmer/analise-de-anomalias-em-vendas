"""Experimental daily reference with observed promotion context.

The model estimates associations, not causal effects. Only the training block
fits coefficients and residual scale; only calibration scores select the alert
threshold. The calendar is an independent input and never supplies anomaly labels.
"""

import math

import numpy as np
import pandas as pd

from src.analysis.historical_detection import (
    MAD_SCALE,
    SCORE_COLUMNS,
    apply_threshold,
    calibrate_threshold,
    temporal_split,
)


CONTEXT_COLUMNS = ("promocao",)
FEATURE_NAMES = (
    "intercept", "trend_per_week",
    "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
    *CONTEXT_COLUMNS,
)
LIMITATIONS = [
    "Experimento contextual: os coeficientes descrevem associações e não comprovam causas comerciais.",
    "O calendário precisa registrar presença e ausência dos acontecimentos; ausência de informação não é ausência do evento.",
    "A referência usa dia da semana, tendência linear no log e um efeito multiplicativo de promoção; não modela todas as interações.",
    "Os limites são operacionais, não intervalos de confiança nem uma previsão validada de vendas futuras.",
    "A referência sem contexto apenas zera os indicadores no mesmo modelo; não é um efeito causal estimado.",
]


def _positive(value, name, allow_zero=False):
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{name} deve ser finito e {'não negativo' if allow_zero else 'positivo'}.")
    return number


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} deve ser um inteiro positivo.")
    return int(value)


def _dates(values, name):
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{name} contém datas ausentes ou inválidas.")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    dates = dates.dt.normalize()
    if dates.duplicated().any():
        raise ValueError(f"{name} precisa de uma linha por data.")
    return dates


def _prepare(series, calendar):
    frame = pd.DataFrame(series).copy()
    frame.attrs.update(getattr(series, "attrs", {}))
    if not {"data", "valor"}.issubset(frame.columns) or frame.empty:
        raise ValueError("A série contextual precisa de data e valor, com ao menos uma linha.")
    if frame.attrs.get("frequency", "D") != "D":
        raise ValueError("O experimento contextual aceita somente séries diárias.")
    frame["data"] = _dates(frame["data"], "A série")
    frame["valor"] = pd.to_numeric(frame["valor"], errors="coerce")
    if np.isinf(frame["valor"].to_numpy(dtype=float)).any():
        raise ValueError("A série contextual contém valores infinitos.")
    frame = frame.sort_values("data", kind="stable").reset_index(drop=True)
    expected_dates = pd.date_range(frame["data"].iloc[0], frame["data"].iloc[-1], freq="D")
    if not np.array_equal(expected_dates.to_numpy(), frame["data"].to_numpy()):
        raise ValueError("Use um calendário diário regular, preservando as datas sem observação.")
    if "split" not in frame:
        frame = temporal_split(frame)
    labels = list(dict.fromkeys(frame["split"].tolist()))
    ranks = frame["split"].map({"train": 0, "calibration": 1, "test": 2})
    if labels != ["train", "calibration", "test"] or ranks.isna().any() or not ranks.is_monotonic_increasing:
        raise ValueError("São necessários blocos distintos de treino, calibração e teste, nessa ordem.")

    context = pd.DataFrame(calendar).copy()
    required = {"data", *CONTEXT_COLUMNS}
    if not required.issubset(context.columns):
        raise ValueError("O calendário precisa de data e promocao.")
    # Select the documented features explicitly; extra fields (including labels)
    # cannot become regressors through an accidental merge or numeric selection.
    context = context[["data", *CONTEXT_COLUMNS]]
    context["data"] = _dates(context["data"], "O calendário")
    for column in CONTEXT_COLUMNS:
        valid = context[column].dropna().map(
            lambda value: isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating))
            and value in (0, 1)
        )
        if not valid.all():
            raise ValueError(f"{column} deve conter booleanos, 0/1 ou valores ausentes.")
        context[column] = context[column].astype("boolean")
    # Existing same-named columns in series are not a substitute for the separate
    # independently supplied calendar.
    frame = frame.drop(columns=list(CONTEXT_COLUMNS), errors="ignore")
    attrs = dict(frame.attrs)
    frame = frame.merge(context, on="data", how="left", validate="one_to_one", sort=False)
    frame.attrs.update(attrs)
    frame["context_available"] = frame[list(CONTEXT_COLUMNS)].notna().all(axis=1)
    return frame


def _design(frame, origin):
    elapsed_weeks = (frame["data"] - origin).dt.days.to_numpy(dtype=float) / 7.0
    weekday = frame["data"].dt.dayofweek.to_numpy()
    return np.column_stack([
        np.ones(len(frame)), elapsed_weeks,
        *[(weekday == day).astype(float) for day in range(1, 7)],
        *[frame[column].to_numpy(dtype=float, na_value=np.nan) for column in CONTEXT_COLUMNS],
    ])


def _robust_fit(x, y, iterations=60):
    """Huber iteratively reweighted least squares with MAD scale, NumPy only."""
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    completed = 0
    for completed in range(1, iterations + 1):
        residuals = y - x @ coefficients
        centered = residuals - np.median(residuals)
        sigma = MAD_SCALE * float(np.median(np.abs(centered)))
        if sigma < 1e-10:
            break
        # MAD is centered to estimate scale; Huber weights use the residual
        # around the fitted regression, not its median-centered counterpart.
        distance = np.abs(residuals) / (1.345 * sigma)
        weights = np.ones(len(y))
        outside = distance > 1.0
        weights[outside] = 1.0 / distance[outside]
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root_weights[:, None], y * root_weights, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return coefficients, completed


def _unavailable(frame, reason, model=None):
    scored = frame.copy()
    for column in SCORE_COLUMNS:
        scored[column] = False if column in {"flag", "eligible"} else np.nan
    scored["reason"] = reason
    scored["expected_without_context"] = np.nan
    scored["context_factor"] = np.nan
    scored.attrs.update(frame.attrs)
    return {"available": False, "reason": reason, "scored": scored,
            "model": model or {}, "calibration": None, "limitations": list(LIMITATIONS)}


def analyze_contextual_series(
    series, calendar, *, min_train=56, min_context_occurrences=3,
    scale_floor_pct=0.05, absolute_floor=1.0, minimum_threshold=3.0,
):
    """Fit a fixed daily reference on train and calibrate on calibration only.

    Inputs: a regular series with ``data``, ``valor`` and optional chronological
    ``split``; an independent calendar with nullable boolean ``promocao``.
    Unknown dates/context are never converted to no-event days.
    The log fit uses positive training observations. Scoring uses level
    residuals, so an observed complete zero is an evaluable total drop;
    negative or missing observations remain ineligible.
    The return dict contains ``available``, ``reason``, ``scored``, ``model``,
    ``calibration`` and ``limitations``. The caller evaluates labels separately.
    """
    min_train = _count(min_train, "O treino mínimo")
    min_context_occurrences = _count(min_context_occurrences, "O suporte mínimo de contexto")
    floor_pct = _positive(scale_floor_pct, "O piso relativo", allow_zero=True)
    floor_abs = _positive(absolute_floor, "O piso absoluto")
    minimum = _positive(minimum_threshold, "O limiar mínimo")
    frame = _prepare(series, calendar)
    values = frame["valor"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    complete = frame["completo"].fillna(False).to_numpy(dtype=bool) if "completo" in frame else finite
    known_context = frame["context_available"].to_numpy(dtype=bool)
    positive_training_values = finite & (values > 0) & complete
    valid_scoring_values = finite & (values >= 0) & complete
    train_mask = frame["split"].eq("train").to_numpy() & positive_training_values & known_context
    n_train = int(train_mask.sum())
    if n_train < max(min_train, len(FEATURE_NAMES) + 1):
        return _unavailable(frame, "insufficient_training_history")
    support = {}
    for column in CONTEXT_COLUMNS:
        flags = frame.loc[train_mask, column].astype(int)
        support[column] = {"present": int(flags.sum()), "absent": int((flags == 0).sum())}
        if min(support[column].values()) < min_context_occurrences:
            return _unavailable(frame, "insufficient_context_history:" + column)

    origin = frame.loc[train_mask, "data"].iloc[0]
    x = _design(frame, origin)
    x_train = x[train_mask]
    rank = int(np.linalg.matrix_rank(x_train))
    if rank != len(FEATURE_NAMES) or np.linalg.cond(x_train) > 1e8:
        return _unavailable(frame, "rank_deficient_context")
    y_train = np.log(values[train_mask])
    coefficients, iterations = _robust_fit(x_train, y_train)
    residuals = y_train - x_train @ coefficients
    log_mad = float(np.median(np.abs(residuals - np.median(residuals))))
    residual_fraction = math.expm1(MAD_SCALE * log_mad)
    relative_scale = max(residual_fraction, floor_pct)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.exp(x @ coefficients)
        x_no_context = x.copy()
        x_no_context[:, -len(CONTEXT_COLUMNS):] = 0.0
        expected_without_context = np.exp(x_no_context @ coefficients)
        context_factor = np.exp(x[:, -len(CONTEXT_COLUMNS):] @ coefficients[-len(CONTEXT_COLUMNS):])
    reference_valid = np.isfinite(expected) & (expected > 0)
    scale = np.maximum(expected * relative_scale, floor_abs)
    eligible = valid_scoring_values & known_context & reference_valid & ~frame["split"].eq("train").to_numpy()
    score = np.full(len(frame), np.nan)
    score[eligible] = np.abs(values[eligible] - expected[eligible]) / scale[eligible]
    reasons = np.select(
        [~finite, finite & (values < 0), ~complete, ~known_context, ~reference_valid,
         frame["split"].eq("train").to_numpy()],
        ["missing_observation", "negative_observation", "partial_period", "missing_context",
         "invalid_reference", "training_fit_only"], default="ready",
    )
    scored = frame.assign(
        expected=expected, scale=scale, score=score, lower=np.nan, upper=np.nan,
        flag=False, eligible=eligible, n_history=n_train, reason=reasons,
        decision_threshold=np.nan, expected_without_context=expected_without_context,
        context_factor=context_factor,
    )
    scored.attrs.update(frame.attrs)
    scored.attrs.update(
        baseline="training_only_log_linear_contextual", context_columns=list(CONTEXT_COLUMNS),
        limits_type="empirical_decision_limits_not_confidence_intervals",
        prediction_type="conditional_reference_not_forecast",
    )
    model = {
        "method": "huber_irls_log_linear", "fitted_on": "train_only", "frozen": True,
        "training_domain": "positive_observations_for_log_fit",
        "scoring_domain": "nonnegative_observations",
        "origin": origin.date().isoformat(), "n_train": n_train,
        "training_start": frame.loc[train_mask, "data"].min().date().isoformat(),
        "training_end": frame.loc[train_mask, "data"].max().date().isoformat(),
        "feature_names": list(FEATURE_NAMES),
        "coefficients": dict(zip(FEATURE_NAMES, coefficients.astype(float).tolist())),
        "context_factors": {column: float(np.exp(coefficients[FEATURE_NAMES.index(column)]))
                            for column in CONTEXT_COLUMNS},
        "context_support": support, "matrix_rank": rank, "irls_iterations": iterations,
        "training_log_residual_mad": log_mad, "relative_scale": relative_scale,
        "scale_floor_pct": floor_pct, "absolute_floor": floor_abs,
        "interpretation": "associations_not_causal_effects",
    }
    try:
        calibration = calibrate_threshold(scored, method="robust", minimum=minimum)
    except ValueError:
        return _unavailable(frame, "insufficient_calibration_history", model)
    scored = apply_threshold(scored, calibration)
    return {"available": True, "reason": "ready", "scored": scored, "model": model,
            "calibration": calibration, "limitations": list(LIMITATIONS)}
