"""Resolve monthly investigation references without guessing among several alerts."""

import pandas as pd

from src.agent.period_parser import extract_periods_from_text


def comparison_context(report):
    periods = report.get("periodos", {})
    previous, current = periods.get("periodo_anterior"), periods.get("periodo_atual")
    if not previous or not current:
        return None
    return {"previous": previous, "current": current,
            "window_note": report.get("contexto_da_base", {}).get("comparison_window_definition")}


def resolve_comparison(prompt, data, context=None):
    """Explicit months override context; one month uses its observed predecessor."""
    available = set(data["data"].dt.to_period("M").astype(str))
    explicit = list(dict.fromkeys(extract_periods_from_text(prompt, data)))
    if len(explicit) == 2:
        pair, origin = sorted(explicit), "explicit"
    elif len(explicit) == 1:
        current = explicit[0]
        pair, origin = [str(pd.Period(current, freq="M") - 1), current], "previous_month"
    elif explicit:
        return {"periods": None, "reason": "too_many", "origin": None}
    elif context and context.get("previous") and context.get("current"):
        pair, origin = [context["previous"], context["current"]], "context"
    else:
        return {"periods": None, "reason": "missing", "origin": None}
    if any(period not in available for period in pair) or pair[0] >= pair[1]:
        return {"periods": None, "reason": "unavailable", "origin": origin}
    return {"periods": pair, "reason": None, "origin": origin}
