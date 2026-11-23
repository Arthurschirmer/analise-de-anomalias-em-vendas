"""Currency labels for calculated values, without exchange-rate conversion."""

import math
import re
from collections.abc import Mapping


_SYMBOLS = {"BRL": "R$", "GBP": "£", "USD": "US$", "EUR": "€"}


def _metadata(context):
    attributes = getattr(context, "attrs", None)
    if isinstance(attributes, Mapping):
        context = attributes
    if not isinstance(context, Mapping):
        return {}
    result = dict(context)
    for name in ("dataset_context", "contexto_da_base"):
        nested = context.get(name)
        if isinstance(nested, Mapping):
            result.update(nested)
    return result


def currency_symbol(context=None):
    """Return the source currency symbol; older datasets default to Brazilian reais."""
    metadata = _metadata(context)
    code = str(metadata.get("currency") or "").strip().upper()
    if code:
        return _SYMBOLS.get(code, code) if re.fullmatch(r"[A-Z]{3}", code) else "R$"
    symbol = metadata.get("currency_symbol")
    return symbol if symbol in _SYMBOLS.values() else "R$"


def format_currency(value, context=None):
    """Format a source amount with Portuguese separators, preserving its magnitude."""
    if isinstance(value, bool):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    if not math.isfinite(numeric):
        return "—"
    amount = f"{numeric:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{currency_symbol(context)} {amount}"


def currency_axis_label(context=None):
    """Return a revenue axis label in the original currency."""
    return f"Faturamento ({currency_symbol(context)})"
