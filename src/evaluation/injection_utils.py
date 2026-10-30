"""Integer daily intervention allocation shared by offline experiments."""
import numpy as np
import pandas as pd


def allocate_integer_quantities(quantities, factor, rng):
    """Round the basket total once, then allocate by largest remainders.

    The prices of existing rows are preserved by callers. Seeded ties prevent
    input order from systematically choosing which products change.
    """
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("O fator da injeção deve ser finito e não negativo.")
    values = quantities.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.rint(values), rtol=0, atol=1e-8)):
        raise ValueError("As injeções de volume exigem quantidades inteiras não negativas.")
    desired = values * factor
    allocated = np.floor(desired).astype(np.int64)
    target = int(round(float(values.sum()) * factor))
    remaining = target - int(allocated.sum())
    if remaining:
        remainder = desired - allocated
        order = np.lexsort((rng.random(len(values)), -remainder))
        allocated[order[:remaining]] += 1
    return pd.Series(allocated, index=quantities.index, dtype=float), target

