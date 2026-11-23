"""Readable, deterministic facts for an already calculated historical alert."""

import math


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def explain_alert(row, calibration, metric="faturamento"):
    observed, expected = _finite(row.get("valor")), _finite(row.get("expected"))
    difference = observed - expected if observed is not None and expected is not None else None
    percentage = difference / expected * 100 if difference is not None and expected > 0 else None
    score, threshold = _finite(row.get("score")), _finite(calibration.get("threshold"))
    history_count = _finite(row.get("n_history"))
    return {"period": str(row.get("periodo", "")), "metric": metric, "observed": observed,
            "expected": expected, "difference": difference, "percentage": percentage,
            "score": score, "threshold": threshold, "previous_observations": int(history_count) if history_count is not None else 0,
            "direction": "indisponível" if difference is None else "abaixo" if difference < 0 else "acima" if difference > 0 else "igual",
            "eligible": bool(row.get("eligible", False)), "flagged": bool(row.get("flag", False)),
            "next_checks": ["Confira os registros e a cobertura das datas do período.",
                            "Examine a quantidade e os grupos com informação disponível.",
                            "Confirme promoções, estoque ou eventos externos somente com fontes adicionais."]}
