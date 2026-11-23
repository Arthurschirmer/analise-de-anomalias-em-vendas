"""Reproducible sales, external context and a separate simulation answer key.

Promotions are *assumptions imposed by this generator*. Their
effects are not estimates of real commercial causality.
Only an extra, undisclosed drop in the test block is labelled an anomaly under
the conditional evaluation. Ordinary promotion days remain expected cases.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.loader import STANDARD_COLUMNS


PROMOTION_OFFSETS = (5, 6, 7, 22, 23)
CONTEXT_CYCLE_DAYS = 37
PROMOTION_QUANTITY_FACTOR = 1.45
PROMOTION_DISCOUNT = 0.10
UNEXPECTED_QUANTITY_FACTOR = 0.50

PRODUCTS = (
    ("Notebook", "Eletronicos", "Sul", "Vendedor A", 30, 3000.0),
    ("Monitor", "Eletronicos", "Sul", "Vendedor B", 50, 850.0),
    ("Cadeira", "Casa", "Sudeste", "Vendedor C", 80, 450.0),
    ("Mesa", "Casa", "Sudeste", "Vendedor D", 60, 700.0),
)


@dataclass(frozen=True)
class ContextualDemoDataset:
    """The calendar is model input; truth/metadata are evaluation provenance."""

    sales: pd.DataFrame
    calendar: pd.DataFrame
    truth: pd.DataFrame
    metadata: dict


def _allocate_quantities(base, desired_total):
    """Preserve an integer daily total using proportional largest remainders."""
    weights = np.asarray(base, dtype=float)
    ideal = weights / weights.sum() * int(desired_total)
    allocated = np.floor(ideal).astype(int)
    remaining = int(desired_total) - int(allocated.sum())
    order = np.argsort(-(ideal - allocated), kind="stable")
    allocated[order[:remaining]] += 1
    return allocated


def _unexpected_positions(calendar, test_start):
    """Choose dates from the calendar, never from sales or detector results."""
    promo = calendar["promocao"].to_numpy(dtype=bool)
    positions = np.arange(len(calendar))
    in_test = positions >= test_start
    test_length = len(calendar) - test_start
    normal = ~promo
    normal_starts = positions[:-1][
        in_test[:-1] & normal[:-1] & normal[1:]
    ]
    first = int(normal_starts[np.argmin(abs(normal_starts - (test_start + .15 * test_length)))])
    events = {first: "unexpected_drop_normal", first + 1: "unexpected_drop_normal"}
    for index, fraction in enumerate((.40, .60, .80), start=1):
        eligible = positions[in_test & promo & ~np.isin(positions, list(events))]
        chosen = int(eligible[np.argmin(abs(eligible - (test_start + fraction * test_length)))])
        events[chosen] = f"unexpected_drop_promotion_{index}"
    return events


def generate_contextual_sales(seed=42, days=420, start="2023-01-02"):
    """Return four-product daily sales, context-only calendar and daily truth.

    The default blocks have 210 training, 105 calibration and 105 test days.
    A fixed 37-day calendar distributes promotions across weekdays in every
    block. Promotion multiplies volume by 1.45 and applies a measured 10%
    discount. Five test days receive an additional .50 volume factor:
    two consecutive days without promotion and three distinct promotion days.
    These surprises never enter the calendar.

    Daily quantities are rounded once and allocated proportionally to products.
    The answer key records nominal and realized effects after integer rounding.
    Neither answer-key columns nor metadata are attached to predictor attrs.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("A semente deve ser um inteiro entre 0 e 2^32 - 1.")
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or days < 280:
        raise ValueError("A demonstração precisa de pelo menos 280 dias completos.")
    try:
        first_date = pd.Timestamp(start)
    except (TypeError, ValueError):
        raise ValueError("Informe uma data inicial válida.") from None
    if pd.isna(first_date) or first_date.tz is not None:
        raise ValueError("Informe uma data inicial válida, sem fuso horário.")
    dates = pd.date_range(first_date.normalize(), periods=int(days), freq="D")
    positions = np.arange(int(days))
    phase = positions % CONTEXT_CYCLE_DAYS
    calendar = pd.DataFrame({
        "data": dates,
        "promocao": np.isin(phase, PROMOTION_OFFSETS),
    })
    train_end = int(days) // 2
    calibration_end = train_end + int(days) // 4
    splits = np.where(positions < train_end, "train",
                      np.where(positions < calibration_end, "calibration", "test"))
    events = _unexpected_positions(calendar, calibration_end)
    rng = np.random.default_rng(int(seed))
    weekly = np.array([1.00, 1.05, 1.10, 1.05, 1.15, .85, .65])
    noise = np.exp(rng.normal(0, .012, size=(int(days), len(PRODUCTS))))
    base_volumes = np.array([product[4] for product in PRODUCTS], dtype=float)
    prices = np.array([product[5] for product in PRODUCTS], dtype=float)
    rows, truth_rows = [], []
    for i, day in enumerate(dates):
        promo = bool(calendar.at[i, "promocao"])
        discount = PROMOTION_DISCOUNT if promo else 0.0
        promotion_factor = PROMOTION_QUANTITY_FACTOR if promo else 1.0
        context_factor = promotion_factor
        context_name = "promotion" if promo else "normal"
        baseline = np.maximum(1, np.rint(
            base_volumes * weekly[day.dayofweek] * (1 + .00035 * i) * noise[i]
        ).astype(int))
        expected_total = int(np.rint(baseline.sum() * context_factor))
        contextual = _allocate_quantities(baseline, expected_total)
        is_anomaly = i in events
        extra_factor = UNEXPECTED_QUANTITY_FACTOR if is_anomaly else 1.0
        actual = _allocate_quantities(contextual, int(np.rint(contextual.sum() * extra_factor)))
        for product, quantity in zip(PRODUCTS, actual):
            name, category, region, seller, _, price = product
            rows.append({
                "data": day, "produto": name, "categoria": category,
                "regiao": region, "vendedor": seller, "quantidade": int(quantity),
                "preco_unitario": price, "desconto": discount,
                "faturamento": float(quantity * price * (1 - discount)),
            })
        baseline_revenue = float(baseline @ prices)
        context_revenue = float(contextual @ prices * (1 - discount))
        actual_revenue = float(actual @ prices * (1 - discount))
        truth_rows.append({
            "data": day, "day_index": i, "split": str(splits[i]),
            "expected_context_event": context_name,
            "is_anomaly": bool(is_anomaly), "event_id": events.get(i, ""),
            "baseline_quantity": int(baseline.sum()),
            "context_quantity": int(contextual.sum()), "quantity": int(actual.sum()),
            "baseline_revenue": baseline_revenue,
            "context_revenue": context_revenue, "revenue": actual_revenue,
            "nominal_context_quantity_change_pct": 100 * (context_factor - 1),
            "realized_context_quantity_change_pct": 100 * (contextual.sum() / baseline.sum() - 1),
            "nominal_context_revenue_change_pct": 100 * (context_factor * (1 - discount) - 1),
            "realized_context_revenue_change_pct": 100 * (context_revenue / baseline_revenue - 1),
            "nominal_unexpected_quantity_change_pct": 100 * (extra_factor - 1),
            "realized_unexpected_quantity_change_pct": 100 * (actual.sum() / contextual.sum() - 1),
            "realized_unexpected_revenue_change_pct": 100 * (actual_revenue / context_revenue - 1),
        })
    sales = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    truth = pd.DataFrame(truth_rows)
    metadata = {
        "source": "synthetic_contextual_sales", "version": 2, "context_columns": ["promocao"],
        "seed": int(seed), "days": int(days),
        "start": dates[0].date().isoformat(), "end": dates[-1].date().isoformat(),
        "splits": {label: {"start": dates[splits == label][0].date().isoformat(),
                           "end": dates[splits == label][-1].date().isoformat(),
                           "count": int((splits == label).sum())}
                   for label in ("train", "calibration", "test")},
        "context_cycle_days": CONTEXT_CYCLE_DAYS,
        "promotion_offsets": list(PROMOTION_OFFSETS),
        "promotion_quantity_factor": PROMOTION_QUANTITY_FACTOR,
        "promotion_discount": PROMOTION_DISCOUNT,
        "unexpected_quantity_factor": UNEXPECTED_QUANTITY_FACTOR,
        "unexpected_days": int(truth["is_anomaly"].sum()),
        "ground_truth_definition": "Only extra unexpected test drops are positive conditional anomalies; scheduled context effects are expected.",
        "causality_note": "Promotion effects are assumptions imposed by the generator, not evidence of causality in real data.",
        "rounding": "Round each daily total once (nearest integer, ties to even), allocate by largest remainders, ties by product order.",
    }
    return ContextualDemoDataset(sales=sales, calendar=calendar, truth=truth, metadata=metadata)
