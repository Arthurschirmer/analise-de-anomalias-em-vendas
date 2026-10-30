"""Persist human judgements and construct auditable stratified review panels.

A sampled panel is an assessment design, never automatically generated truth.
No point is labelled normal merely because the detector did not flag it.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

JUDGEMENTS = ["Anomalia confirmada", "Comportamento esperado", "Inconclusivo"]
STRATA = ("flagged", "unflagged")


def _valid_scope(scope):
    if not isinstance(scope, str) or len(scope) != 64 or any(char not in "0123456789abcdef" for char in scope):
        raise ValueError("Identificador da análise inválido.")


def review_scope(data, configuration):
    digest = hashlib.sha256(pd.util.hash_pandas_object(
        data.sort_values(list(data.columns)).reset_index(drop=True), index=False).values.tobytes())
    digest.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def read_reviews(scope, output_dir="outputs/reviews"):
    _valid_scope(scope)
    path = Path(output_dir) / (scope + ".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_review(scope, period, judgement, reviewer, notes="", output_dir="outputs/reviews", *, metadata=None):
    if judgement not in JUDGEMENTS or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Informe o responsável e um parecer válido.")
    _valid_scope(scope)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Os metadados da revisão devem ser um objeto.")
    # Validate metadata before mutating an existing review.
    try:
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        raise ValueError("Os metadados da revisão devem ser serializáveis e finitos.") from None
    result = read_reviews(scope, output_dir)
    previous = result.get(period, {})
    entry = {"periodo": (metadata or {}).get("periodo", period), "judgement": judgement,
             "reviewer": reviewer.strip(), "notes": notes,
             "reviewed_at": datetime.now(timezone.utc).isoformat()}
    if metadata is not None:
        entry["metadata"] = dict(metadata)
        entry["review_id"] = period
    if previous:
        entry["history"] = [*previous.get("history", []),
                            {key: value for key, value in previous.items() if key != "history"}]
    result[period] = entry
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (scope + ".json")
    temp = out / (scope + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)
    return result


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Tipo não serializável: {type(value).__name__}")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} deve ser um inteiro positivo.")
    return int(value)


def create_review_sample(scored, sample_size=30, seed=42, blind=True):
    """Sample eligible test points within both detector-result strata.

    The sampling frame is the eligible test set, not every transaction or every
    date. Equal allocation improves inspection of rare alerts; design weights
    correct this allocation when a fully judged panel is summarised.
    """
    sample_size = _positive_integer(sample_size, "O tamanho da amostra")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 or seed > 2**32 - 1:
        raise ValueError("A semente deve ser um inteiro entre 0 e 2^32-1.")
    if not isinstance(blind, (bool, np.bool_)):
        raise ValueError("A opção de revisão cega deve ser booleana.")
    required = {"periodo", "split", "eligible", "flag"}
    if not required.issubset(scored.columns):
        raise ValueError("A amostragem precisa de periodo, split, eligible e flag.")
    frame = scored.loc[scored["split"].eq("test") & scored["eligible"].fillna(False)].copy()
    if frame["flag"].isna().any() or not frame["flag"].map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("A sinalização dos pontos elegíveis deve ser booleana.")
    identity_columns = [key for key in ["dimensao", "segmento", "periodo"] if key in frame]
    identity = frame[identity_columns].astype(str)
    if identity.duplicated().any():
        raise ValueError("A amostra exige um ponto único por período e segmento.")
    frame["review_id"] = [
        hashlib.sha256(json.dumps(row, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
        for row in identity.itertuples(index=False, name=None)
    ]
    frame["stratum"] = np.where(frame["flag"], "flagged", "unflagged")
    frame = frame.sort_values("review_id", kind="stable").reset_index(drop=True)
    counts = {stratum: int(frame["stratum"].eq(stratum).sum()) for stratum in STRATA}
    present_strata = sum(count > 0 for count in counts.values())
    target = min(len(frame), sample_size)
    if target < present_strata:
        raise ValueError("Use pelo menos dois pontos para incluir sinalizados e não sinalizados.")
    flagged_n = min(counts["flagged"], (target + 1) // 2)
    unflagged_n = min(counts["unflagged"], target - flagged_n)
    remainder = target - flagged_n - unflagged_n
    extra = min(remainder, counts["flagged"] - flagged_n)
    flagged_n += extra
    unflagged_n += remainder - extra
    allocation = {"flagged": flagged_n, "unflagged": unflagged_n}
    rng = np.random.default_rng(int(seed))
    selected = []
    for stratum in STRATA:
        group = frame.loc[frame["stratum"].eq(stratum)]
        amount = allocation[stratum]
        if not amount:
            continue
        positions = rng.choice(len(group), size=amount, replace=False)
        part = group.iloc[positions].copy()
        part["stratum_population"] = counts[stratum]
        part["stratum_sample"] = amount
        part["inclusion_probability"] = amount / counts[stratum]
        part["sample_weight"] = counts[stratum] / amount
        selected.append(part)
    extra_columns = ["stratum_population", "stratum_sample", "inclusion_probability", "sample_weight"]
    result = (pd.concat(selected, ignore_index=True) if selected else
              frame.assign(**{key: pd.Series(dtype=float) for key in extra_columns}))
    if len(result):
        result = result.iloc[rng.permutation(len(result))].reset_index(drop=True)
    result["review_order"] = np.arange(1, len(result) + 1)
    # Include the complete frame and all decisions, not just sampled points, in
    # the design identity. A changed score/flag cannot silently reuse a panel.
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    digest.update(json.dumps({"sample_size": sample_size, "seed": int(seed), "blind": bool(blind)},
                             sort_keys=True).encode("utf-8"))
    result.attrs.update(
        sample_id=digest.hexdigest(), population_counts=counts, sample_counts=allocation,
        population_size=len(frame), requested_size=sample_size, seed=int(seed), blind=bool(blind),
        sampling_method="stratified_simple_random_without_replacement",
        review_population="eligible_test_points",
        limitation="Pareceres são humanos; pontos sem alerta não recebem rótulo automático de normalidade.",
    )
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_review_sample(sample, reviews):
    """Describe coverage and, only for complete panels, design-weighted estimates.

    Inconclusive or absent judgements prevent numeric performance claims. Even
    for complete panels, weighted estimates describe this finite test sampling
    frame and depend on human judgements; no confidence interval is implied.
    """
    required = {"review_id", "stratum", "flag", "sample_weight", "stratum_population"}
    if not required.issubset(sample.columns):
        raise ValueError("O painel não contém as colunas de amostragem necessárias.")
    if sample["review_id"].duplicated().any():
        raise ValueError("O painel contém identificadores de revisão duplicados.")
    if not sample["stratum"].isin(STRATA).all():
        raise ValueError("O painel contém um estrato desconhecido.")
    if len(sample) and not sample["flag"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("A sinalização do painel deve ser booleana.")
    if len(sample) and not sample["stratum"].eq(np.where(sample["flag"], "flagged", "unflagged")).all():
        raise ValueError("O estrato não corresponde à sinalização do ponto.")
    population_counts = dict(sample.attrs.get("population_counts", {}))
    for stratum in STRATA:
        values = sample.loc[sample["stratum"].eq(stratum), "stratum_population"]
        if len(values) and values.nunique() != 1:
            raise ValueError("População inconsistente em um estrato da amostra.")
        population_counts.setdefault(stratum, int(values.iloc[0]) if len(values) else 0)
    population_counts = {key: int(population_counts[key]) for key in STRATA}
    coverage = {}
    observed = dict.fromkeys(["tp", "fp", "fn", "tn"], 0)
    weighted = dict.fromkeys(["tp", "fp", "fn", "tn"], 0.0)
    for stratum in STRATA:
        part = sample.loc[sample["stratum"].eq(stratum)]
        counts = {"population": population_counts[stratum], "sampled": len(part),
                  "reviewed": 0, "definitive": 0, "inconclusive": 0, "missing": 0}
        for row in part.to_dict(orient="records"):
            review = reviews.get(row["review_id"], {})
            saved_sample_id = review.get("metadata", {}).get("sample_id")
            current_sample_id = sample.attrs.get("sample_id")
            # Row identities intentionally remain stable across panels. An
            # explicitly different panel provenance must not become fresh truth.
            stale_panel = (saved_sample_id is not None and current_sample_id is not None
                           and saved_sample_id != current_sample_id)
            judgement = None if stale_panel else review.get("judgement")
            if judgement not in JUDGEMENTS:
                counts["missing"] += 1
                continue
            counts["reviewed"] += 1
            if judgement == "Inconclusivo":
                counts["inconclusive"] += 1
                continue
            counts["definitive"] += 1
            anomaly = judgement == "Anomalia confirmada"
            flag = bool(row["flag"])
            cell = "tp" if flag and anomaly else "fp" if flag else "fn" if anomaly else "tn"
            observed[cell] += 1
            weight = float(row["sample_weight"])
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("Os pesos amostrais devem ser positivos e finitos.")
            weighted[cell] += weight
        coverage[stratum] = counts
    total = lambda field: sum(row[field] for row in coverage.values())
    complete = bool(len(sample)) and total("definitive") == len(sample)
    covered_strata = all(not pop or coverage[key]["sampled"] for key, pop in population_counts.items())
    census = complete and covered_strata and len(sample) == sum(population_counts.values())
    metrics = None
    status = ("empty" if not len(sample) else "incomplete" if total("missing") else
              "inconclusive" if total("inconclusive") else "missing_stratum" if not covered_strata else "complete")
    if complete and covered_strata:
        tp, fp, fn, tn = (weighted[key] for key in ["tp", "fp", "fn", "tn"])
        metrics = {
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "specificity": _ratio(tn, tn + fp), "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
            "basis": "eligible_test_census" if census else "stratified_sample_weighted_estimate",
            "is_estimate": not census, "weighted_confusion": weighted,
        }
    return {
        "status": status, "sample_id": sample.attrs.get("sample_id"),
        "population_size": sum(population_counts.values()), "sampled": len(sample),
        "reviewed": total("reviewed"), "definitive": total("definitive"),
        "missing": total("missing"), "inconclusive": total("inconclusive"),
        "coverage_by_stratum": coverage, "observed_reviewed_confusion": observed,
        "metrics": metrics, "is_census": census,
        "limitations": [
            "A matriz observada descreve apenas os pareceres conclusivos disponíveis.",
            "Métricas ficam suspensas enquanto houver pareceres ausentes ou inconclusivos.",
            "Estimativas ponderadas descrevem apenas os pontos elegíveis do teste, não toda a base.",
            "Não há intervalo de confiança; erros e divergências dos revisores afetam os resultados.",
            "A revisão pode ser cega aos alertas para reduzir influência da decisão do detector.",
        ],
    }


def save_review_sample(scope, sample, output_dir="outputs/reviews"):
    """Freeze the sampled records and design metadata separately from judgements."""
    _valid_scope(scope)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (scope + "_sample.json")
    # to_json makes unknown numeric values null and timestamps portable.
    records = json.loads(sample.to_json(orient="records", date_format="iso"))
    payload = {"metadata": sample.attrs, "columns": list(sample.columns), "records": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False,
                               default=_json_default), encoding="utf-8")
    return path


def read_review_sample(scope, output_dir="outputs/reviews"):
    _valid_scope(scope)
    path = Path(output_dir) / (scope + "_sample.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = pd.DataFrame(payload["records"], columns=payload["columns"])
    for key in ["data", "inicio_periodo", "fim_periodo"]:
        if key in result:
            result[key] = pd.to_datetime(result[key], errors="coerce")
    result.attrs.update(payload["metadata"])
    return result


