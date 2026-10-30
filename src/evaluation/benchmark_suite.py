"""Declared scenario suite with paired seeds, frozen calibration and partial labels.

All scenarios and intensities are fixed below before execution. This is an
exploratory comparison; the Olist periods have already been viewed in development.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.historical_detection import (
    apply_threshold, build_time_series, calibrate_threshold,
    score_historical_series, temporal_split,
)
from src.evaluation.anomaly_benchmark import _baseline, _metrics, clean_json, controlled_sales
from src.evaluation.injection_utils import allocate_integer_quantities as _allocate_integer_quantities

DEFAULT_SEEDS = (42, 137, 2026)
SCENARIOS = (
    {"name": "normal_control", "type": "normal_control", "intensity": 0., "duration": 0},
    {"name": "mild_drop", "type": "volume_drop", "intensity": .10, "duration": 2},
    {"name": "moderate_drop", "type": "volume_drop", "intensity": .30, "duration": 2},
    {"name": "strong_drop", "type": "volume_drop", "intensity": .60, "duration": 2},
    {"name": "mild_spike", "type": "volume_spike", "intensity": .10, "duration": 2},
    {"name": "moderate_spike", "type": "volume_spike", "intensity": .30, "duration": 2},
    {"name": "strong_spike", "type": "volume_spike", "intensity": .60, "duration": 2},
    {"name": "gradual_drop", "type": "gradual_drop", "intensity": .40, "duration": 14},
    {"name": "persistent_drop", "type": "persistent_drop", "intensity": .25, "duration": 10},
    {"name": "category_hidden_drop", "type": "compensated_segment_drop", "dimension": "categoria",
     "intensity": .35, "duration": 5},
    {"name": "region_hidden_drop", "type": "compensated_segment_drop", "dimension": "regiao",
     "intensity": .35, "duration": 5},
)
DETECTORS = ("previous_week_30pct", "historical_quantile98", "seasonal_trend_mad")


def _validate_seeds(seeds):
    seeds = tuple(seeds)
    if not seeds or len(seeds) > 20 or len(set(seeds)) != len(seeds):
        raise ValueError("Informe de 1 a 20 sementes distintas.")
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
           or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("As sementes devem ser inteiros entre 0 e 2^32 - 1.")
    return tuple(int(seed) for seed in seeds)


def _split_original(data, *, real=False):
    series = build_time_series(data)
    if not real:
        return temporal_split(series)
    if series.empty:
        raise ValueError("O recorte real não contém observações.")
    train_end, calibration_end = pd.Timestamp("2017-08-31"), pd.Timestamp("2017-12-31")
    series["split"] = np.where(series["data"].le(train_end), "train",
                                np.where(series["data"].le(calibration_end), "calibration", "test"))
    counts = series.groupby("split").size()
    if any(counts.get(label, 0) < minimum for label, minimum in
           [("train", 56), ("calibration", 28), ("test", 28)]):
        raise ValueError("A suíte Olist requer histórico de treino até ago/2017, calibração set–dez/2017 e teste em 2018.")
    boundaries = {}
    for label, part in series.groupby("split", sort=False):
        boundaries[label] = {"start": part["data"].min().date().isoformat(),
                             "end": part["data"].max().date().isoformat(), "count": len(part)}
    series.attrs["split_boundaries"] = boundaries
    series.attrs["split_method"] = "declared_calendar_blocks"
    return series


def _align_series(data, reference):
    frame = build_time_series(data)
    frame = frame.set_index("data").reindex(reference["data"]).reset_index()
    attrs = dict(reference.attrs)
    if frame.empty:
        return frame
    frame["periodo"] = frame["data"].dt.strftime("%Y-%m-%d")
    frame["split"] = reference["split"].to_numpy()
    for col in ["observado", "completo"]:
        frame[col] = frame[col].fillna(False).astype(bool)
    frame.attrs.update(attrs)
    return frame


def _calibrations(original):
    scored = score_historical_series(original, trend="linear")
    return {
        "seasonal_trend_mad": calibrate_threshold(scored, method="robust"),
        "historical_quantile98": calibrate_threshold(scored, method="quantile"),
    }


def _scores(changed, original, calibrations):
    series = _align_series(changed, original)
    scored = score_historical_series(series, trend="linear")
    return {
        "previous_week_30pct": _baseline(series),
        "historical_quantile98": apply_threshold(scored, calibrations["historical_quantile98"]),
        "seasonal_trend_mad": apply_threshold(scored, calibrations["seasonal_trend_mad"]),
    }


def _inject_scenario(data, original, specification, seed):
    changed = data.copy(deep=True)
    for col in ["quantidade", "preco_unitario", "faturamento"]:
        changed[col] = changed[col].astype(float)
    spec = dict(specification)
    event = {**spec, "target": None, "start": None, "end": None,
             "actual_positive_days": 0, "total_revenue_preserved": False,
             "daily_effects": [], "status": "normal_control" if not spec["duration"] else "pending",
             "intervention": ("compensated_effective_unit_value" if spec.get("dimension")
                              else "integer_quantity_allocation" if spec["duration"] else "none")}
    if not spec["duration"]:
        return changed, event, set()
    # Placement is seeded and chosen before examining detector results.
    test_dates = original.loc[original["split"].eq("test"), "data"]
    span = (test_dates.max() - test_dates.min()).days + 1
    duration = int(spec["duration"])
    if span < duration + 7:
        raise ValueError("O teste é curto para os cenários de duração prolongada.")
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, span - duration + 1))
    start = pd.Timestamp(test_dates.min()) + pd.Timedelta(days=offset)
    end = start + pd.Timedelta(days=duration - 1)
    event.update(start=start.date().isoformat(), end=end.date().isoformat())
    dates = pd.to_datetime(changed["data"]).dt.normalize()
    dimension = spec.get("dimension")
    target = None
    if dimension:
        train_end = original.loc[original["split"].eq("train"), "data"].max()
        train = data.loc[pd.to_datetime(data["data"]).dt.normalize().le(train_end)]
        ranked = train.groupby(dimension)["faturamento"].sum().sort_values(ascending=False)
        if len(ranked) < 2:
            event["skipped_reason"] = "O cenário compensado requer pelo menos dois segmentos no treino."
            event["status"] = "skipped"
            return changed, event, set()
        target = str(ranked.index[0])
        event["target"] = target
    positive = set()
    for index, day in enumerate(pd.date_range(start, end)):
        mask = dates.eq(day)
        if dimension:
            mask &= changed[dimension].astype(str).eq(target)
        if not mask.any():
            continue
        intensity = spec["intensity"] * ((index + 1) / duration if spec["type"] == "gradual_drop" else 1)
        factor = 1 + intensity if spec["type"] == "volume_spike" else 1 - intensity
        before = changed.loc[mask, "faturamento"].sum()
        before_quantity = float(changed.loc[mask, "quantidade"].sum())
        before_global_revenue = float(changed.loc[dates.eq(day), "faturamento"].sum())
        target_quantity = None
        if dimension:
            # Effective unit values change in the target and compensation basket.
            changed.loc[mask, "preco_unitario"] *= factor
            changed.loc[mask, "faturamento"] *= factor
            lost = before - changed.loc[mask, "faturamento"].sum()
            complement = dates.eq(day) & ~changed[dimension].astype(str).eq(target)
            complement_revenue = changed.loc[complement, "faturamento"].sum()
            if complement_revenue <= 0:
                # No invented compensating record: restore this day's target.
                changed.loc[mask, "preco_unitario"] /= factor
                changed.loc[mask, "faturamento"] /= factor
                continue
            compensation = 1 + lost / complement_revenue
            changed.loc[complement, "preco_unitario"] *= compensation
            changed.loc[complement, "faturamento"] *= compensation
        else:
            old_quantity = changed.loc[mask, "quantidade"].copy()
            new_quantity, target_quantity = _allocate_integer_quantities(old_quantity, factor, rng)
            ratio = new_quantity.div(old_quantity.where(old_quantity.ne(0))).fillna(0)
            changed.loc[mask, "quantidade"] = new_quantity
            changed.loc[mask, "faturamento"] *= ratio
        after = changed.loc[mask, "faturamento"].sum()
        after_quantity = float(changed.loc[mask, "quantidade"].sum())
        after_global_revenue = float(changed.loc[dates.eq(day), "faturamento"].sum())
        event["daily_effects"].append({
            "data": day.date().isoformat(), "nominal_intensity": float(intensity),
            "nominal_factor": float(factor), "original_quantity": before_quantity,
            "target_quantity": target_quantity, "realized_quantity": after_quantity,
            "quantity_change_pct": ((after_quantity / before_quantity - 1) * 100
                                    if before_quantity else None),
            "original_target_revenue": float(before), "realized_target_revenue": float(after),
            "target_revenue_change_pct": (float((after / before - 1) * 100) if before else None),
            "original_global_revenue": before_global_revenue,
            "realized_global_revenue": after_global_revenue,
            "global_revenue_change_pct": ((after_global_revenue / before_global_revenue - 1) * 100
                                          if before_global_revenue else None),
        })
        if not np.isclose(before, after, rtol=0, atol=1e-8):
            positive.add(pd.Timestamp(day))
    event["actual_positive_days"] = len(positive)
    event["total_revenue_preserved"] = bool(dimension and positive)
    event["status"] = "completed" if positive else "no_effect"
    if not positive:
        event["skipped_reason"] = "Nenhum efeito não nulo no faturamento observado; não há positivos injetados para avaliar."
    return changed, event, positive


def _evaluate_channel(changed, original, calibrations, positives, *, complete, metadata):
    scores = _scores(changed, original, calibrations)
    common = original["split"].eq("test")
    for frame in scores.values():
        common &= frame["eligible"]
    rows, points, event_rows = [], [], []
    for name in DETECTORS:
        frame = scores[name].copy()
        frame["evaluation_mask"] = common
        positive = frame["data"].isin(positives)
        frame["is_anomaly"] = positive if complete else positive.map({True: True, False: None})
        frame["label_status"] = np.where(positive, "injected_positive",
                                         "controlled_normal" if complete else "unlabeled_real")
        metrics = _metrics(frame, complete)
        negative_points = (metrics["fp"] + metrics["tn"]) if complete else None
        metrics["false_alert_rate"] = (
            metrics["fp"] / negative_points if negative_points else None)
        metrics["metrics_applicability"] = (
            "known_positive_support" if metrics["known_positive_points"] else "no_evaluable_positives")
        if not metrics["known_positive_points"]:
            # Normal controls and compensated totals have no positive support.
            # They characterize false alerts, not anomaly-recovery performance.
            for key in ["precision", "recall", "f1"]:
                metrics[key] = None
        rows.append({**metadata, "detector": name, **metrics})
        inside = common & positive
        first_flag = frame.loc[inside & frame["flag"], "data"].min()
        first_positive = min(positives) if positives else None
        event_rows.append({
            **metadata, "detector": name,
            "evaluable_positive_days": int(inside.sum()),
            "detected_positive_days": int((inside & frame["flag"]).sum()),
            "event_detected": bool((inside & frame["flag"]).any()) if inside.any() else None,
            "first_detection_delay_days": (
                int((first_flag - first_positive).days)
                if first_positive is not None and pd.notna(first_flag) else None),
        })
        for key, value in metadata.items():
            frame[key] = value
        frame["detector"] = name
        points.append(frame)
    return rows, event_rows, pd.concat(points, ignore_index=True)


def _aggregate(metrics):
    metrics = pd.DataFrame(metrics)
    keys = ["stage", "scenario", "event_type", "intensity", "duration", "scope", "detector"]
    fields = ["precision", "recall", "f1", "injection_recall", "false_alert_rate",
              "unlabeled_alerts", "tp", "fp", "fn", "tn", "evaluated_points", "known_positive_points"]
    records = []
    for group, part in metrics.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, group))
        record["runs"] = len(part)
        for field in fields:
            values = pd.to_numeric(part[field], errors="coerce").dropna()
            record[field + "_n"] = len(values)
            record[field + "_mean"] = float(values.mean()) if len(values) else None
            record[field + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        records.append(record)
    return clean_json(records)


def run_benchmark_suite(output_dir="outputs/benchmark/suite", seeds=DEFAULT_SEEDS, days=420, *,
                        real_data=None, olist_dir=None, scenario_names=None):
    seeds = _validate_seeds(seeds)
    selected = [dict(spec) for spec in SCENARIOS if scenario_names is None or spec["name"] in scenario_names]
    if not selected or (scenario_names is not None and set(scenario_names) - {s["name"] for s in SCENARIOS}):
        raise ValueError("Informe cenários existentes na suíte declarada.")
    if real_data is None and olist_dir is not None:
        from src.data.adapters.olist_adapter import OlistAdapter
        real_data = OlistAdapter(olist_dir).load()
    if real_data is not None:
        real_data = real_data.loc[real_data["data"].between("2017-01-01", "2018-07-31")].copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics, events, points, manifests, injection_effects = [], [], [], [], []
    for seed in seeds:
        datasets = [("controlled", controlled_sales(seed, days), True)]
        if real_data is not None:
            datasets.append(("olist_semisynthetic", real_data, False))
        for stage, data, complete in datasets:
            original = _split_original(data, real=not complete)
            calibrations = _calibrations(original)
            manifests.append({
                "stage": stage, "seed": seed, "input_rows": len(data),
                "input_sha256": hashlib.sha256(data.to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest(),
                "splits": original.attrs["split_boundaries"], "calibration": clean_json(calibrations),
                "source_context": clean_json(data.attrs.get("dataset_context", {})),
                "evaluation_status": "exploratory_previously_viewed_data" if not complete else "controlled_simulation",
            })
            # Target baskets and fits are selected from original train/calibration only.
            for spec in selected:
                index = next(i for i, item in enumerate(SCENARIOS) if item["name"] == spec["name"])
                placement_seed = int(np.random.SeedSequence([seed, index, 17]).generate_state(1)[0])
                changed, event, positive_dates = _inject_scenario(data, original, spec, placement_seed)
                metadata = {"stage": stage, "seed": seed, "scenario": spec["name"],
                            "event_type": spec["type"], "intensity": spec["intensity"],
                            "duration": spec["duration"], "scope": "global",
                            "scenario_status": event["status"], "intervention": event["intervention"]}
                for effect in event["daily_effects"]:
                    injection_effects.append({**metadata, "target": event["target"],
                                              "dimension": spec.get("dimension"), **effect})
                for column in ["quantity_change_pct", "target_revenue_change_pct", "global_revenue_change_pct"]:
                    realized = [effect[column] for effect in event["daily_effects"] if effect[column] is not None]
                    metadata["realized_" + column + "_mean"] = float(np.mean(realized)) if realized else None
                global_positives = set() if spec.get("dimension") else positive_dates
                rows, event_rows, frame = _evaluate_channel(
                    changed, original, calibrations, global_positives, complete=complete, metadata=metadata)
                for row in event_rows:
                    row.update(event_start=event["start"], event_end=event["end"],
                               event_target=event["target"],
                               placement_seed=placement_seed, actual_positive_days=event["actual_positive_days"],
                               scenario_status=event["status"], intervention=event["intervention"],
                               scenario_reason=event.get("skipped_reason"),
                               target_is_total=not bool(spec.get("dimension")))
                    if spec.get("dimension") and event["start"]:
                        local = frame.loc[frame["detector"].eq(row["detector"])]
                        window = local["data"].between(event["start"], event["end"]) & local["evaluation_mask"]
                        row["global_alert_days_during_compensation"] = int(local.loc[window, "flag"].sum())
                metrics.extend(rows)
                events.extend(event_rows)
                points.append(frame)
                if spec.get("dimension") and event["target"] is not None:
                    dimension, target = spec["dimension"], event["target"]
                    original_group = data.loc[data[dimension].astype(str).eq(target)]
                    changed_group = changed.loc[changed[dimension].astype(str).eq(target)]
                    aligned = _align_series(original_group, original)
                    try:
                        segment_fits = _calibrations(aligned)
                        rows, event_rows, frame = _evaluate_channel(
                            changed_group, aligned, segment_fits, positive_dates,
                            complete=complete, metadata={**metadata, "scope": dimension + ":" + target})
                    except ValueError as error:
                        events.append({**metadata, "scope": dimension + ":" + target,
                                       "status": "ineligible", "reason": str(error)})
                    else:
                        for row in event_rows:
                            row.update(event_start=event["start"], event_end=event["end"],
                                       event_target=target, target_is_total=False,
                                       placement_seed=placement_seed, actual_positive_days=event["actual_positive_days"],
                                       scenario_status=event["status"], intervention=event["intervention"])
                        metrics.extend(rows)
                        events.extend(event_rows)
                        points.append(frame)
    aggregate = _aggregate(metrics)
    limitations = [
        "Suíte exploratória; parâmetros e cenários são declarados antes da execução, sem otimização por rótulos do teste.",
        "A Olist já foi observada durante o desenvolvimento; os novos cortes não constituem um conjunto final intocado.",
        "Cortes Olist: treino jan–ago/2017; calibração set–dez/2017; teste jan–jul/2018, sem sobreposição.",
        "Sementes da Olist variam a posição das injeções no mesmo histórico; não são bases reais independentes.",
        "Média e desvio descrevem execuções pareadas entre sementes, sem intervalo de confiança ou teste de superioridade.",
        "Cada média/desvio publica seu número de valores não nulos (metric_n). Controles sem positivos não recebem precisão, recall ou F1; em dados controlados publica-se a fração de falsos alertas entre negativos avaliáveis.",
        "Métricas reais globais continuam sem precisão/F1; somente injeções conhecidas permitem recall parcial.",
        "Nos cenários compensados, o total é preservado e não recebe rótulo positivo; avalia-se o segmento-alvo separado.",
        "Apenas o segmento-alvo declarado é avaliado; outros segmentos compensatórios não são classificados como normais.",
        "Revisão humana permanece necessária para estabelecer anomalias reais e utilidade comercial.",
        "Intensidades nominais de volume se referem à quantidade total do dia, arredondada uma vez e distribuída por maiores restos; o efeito de receita depende dos itens escolhidos.",
        "Os cenários compensados simulam mudanças de valor unitário efetivo no segmento e no complemento, não uma queda observada de pedidos ou volume.",
    ]
    result = {"version": 1, "mode": "offline_no_llm", "seeds": list(seeds), "days": days,
              "configuration": {"frequency": "D", "missing_policy": "unknown", "window": 8,
                                "min_history": 4, "seasonal_period": 7, "trend": "linear",
                                "scale_floor_pct": .05, "absolute_floor": 1, "minimum_threshold": 3,
                                "robust_multiplier": 3, "legacy_quantile": .98,
                                "controlled_split_fractions": [.5, .25, .25],
                                "quantity_allocation": "daily_total_largest_remainders_seeded_ties"},
              "detectors": list(DETECTORS), "scenarios": selected, "runs": clean_json(metrics),
              "aggregate_metrics": aggregate, "event_metrics": clean_json(events),
              "injection_effects": clean_json(injection_effects),
              "manifests": manifests, "limitations": limitations,
              "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    pd.DataFrame(metrics).to_csv(out / "suite_metrics.csv", index=False)
    pd.DataFrame(aggregate).to_csv(out / "suite_aggregate_metrics.csv", index=False)
    pd.DataFrame(events).to_csv(out / "suite_event_metrics.csv", index=False)
    pd.DataFrame(injection_effects).to_csv(out / "suite_injection_effects.csv", index=False)
    pd.concat(points, ignore_index=True).drop(columns="data", errors="ignore").to_csv(out / "suite_point_results.csv", index=False)
    report = ["# Suíte de avaliação dos detectores", "", "Sementes: " + ", ".join(map(str, seeds)),
              "Comparação: variação semanal 30%, histórico com quantil 98% e histórico com calibração robusta.",
              "Limiar robusto: máximo entre 3 e mediana + 3 × 1,4826 MAD dos escores da calibração.",
              "Escala de referência: MAD residual, piso relativo 5% e piso absoluto 1.",
              "Treino, calibração e teste são cronológicos; nenhuma decisão de ajuste usa os rótulos do teste.",
              "Volume: a quantidade total diária desejada é arredondada uma vez, depois distribuída por maiores restos; preços são preservados.",
              "Intensidade nominal refere-se à quantidade. Efeito realizado de receita e quantidades por dia está em suite_injection_effects.csv.",
              "Cenários compensados alteram valor unitário efetivo do segmento e do complemento mantendo a receita total.",
              "", "## Cenários", *["- " + json.dumps(spec, ensure_ascii=False) for spec in selected],
              "", "## Resultados agregados",
              "Cada linha traz média e desvio entre sementes. Ausência de denominador permanece nula."]
    for record in aggregate:
        report.append("- " + json.dumps(clean_json(record), ensure_ascii=False))
    report.extend(["", "## Limitações", *["- " + item for item in limitations]])
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result["outputs"] = {name: str(out / name) for name in [
        "summary.json", "report.md", "suite_metrics.csv", "suite_aggregate_metrics.csv",
        "suite_event_metrics.csv", "suite_point_results.csv", "suite_injection_effects.csv"]}
    result = clean_json(result)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result

