"""Reproducible offline detector benchmark with complete and partial labels."""
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
from src.evaluation.injection_utils import allocate_integer_quantities


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if hasattr(value, "item"):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def controlled_sales(seed=42, days=420):
    if days < 140:
        raise ValueError("Use pelo menos 140 dias para o experimento.")
    rng = np.random.default_rng(seed)
    rows = []
    for index, date in enumerate(pd.date_range("2023-01-02", periods=days)):
        seasonal = [1.1, 1.0, 1.02, 1.05, 1.2, .8, .6][date.weekday()]
        for j, product in enumerate(["A", "B", "C", "D"]):
            quantity = max(1, round((30 + index * .035) * seasonal * rng.normal(1, .025)))
            price = 20 + j * 10
            rows.append({"data": date, "produto": product,
                         "categoria": "Tecnologia" if j < 2 else "Casa",
                         "regiao": "SP" if j % 2 else "PR", "vendedor": "V" + str(j),
                         "quantidade": quantity, "preco_unitario": price,
                         "desconto": 0.0, "faturamento": quantity * price})
    return pd.DataFrame(rows)


def _baseline(series):
    result = series.copy()
    result["expected"] = result["valor"].shift(7)
    result["eligible"] = result["completo"] & result["completo"].shift(7, fill_value=False) & result["expected"].gt(0)
    result["score"] = (result["valor"] - result["expected"]).abs() / result["expected"].abs() * 100
    result["lower"] = result["expected"] * .7
    result["upper"] = result["expected"] * 1.3
    result["flag"] = result["eligible"] & result["score"].gt(30)
    return result


def _fit(series, method="robust"):
    # Threshold fitting sees the original calibration block only.
    scored = score_historical_series(series, trend="linear")
    calibration = calibrate_threshold(scored, method=method)
    return apply_threshold(scored, calibration), calibration


def _inject(data, partitioned, seed=42):
    modified = data.copy(deep=True)
    calendar_dates = pd.to_datetime(modified["data"]).dt.normalize()
    for column in ["preco_unitario", "faturamento", "quantidade"]:
        modified[column] = modified[column].astype(float)
    candidates = partitioned.loc[partitioned["split"].eq("test") & partitioned["completo"], "data"].tolist()
    if len(candidates) < 16:
        raise ValueError("O teste precisa de pelo menos 16 dias observados para as injeções.")
    specs = [
        ("volume_drop", .08, "all", .25),
        ("volume_spike", .30, "all", 2.2),
        ("unit_value_rise", .52, "largest_product", 4.0),
        ("segment_drop", .74, "largest_category", .10),
    ]
    events = []
    label_dates = {}
    rng = np.random.default_rng(seed)
    for index, (kind, fraction, scope, factor) in enumerate(specs, 1):
        start = pd.Timestamp(candidates[int((len(candidates) - 2) * fraction)])
        end = start + pd.Timedelta(days=1)
        mask = calendar_dates.between(start, end)
        target = None
        if scope != "all":
            dim = "produto" if scope == "largest_product" else "categoria"
            target = str(modified.loc[mask].groupby(dim)["faturamento"].sum().idxmax())
            mask &= modified[dim].astype(str).eq(target)
        dates = calendar_dates.loc[mask].drop_duplicates().sort_values()
        event_id = f"E{index:02d}"
        effects = []
        for date in dates:
            day_mask = mask & calendar_dates.eq(date)
            before = float(modified.loc[day_mask, "faturamento"].sum())
            before_quantity = float(modified.loc[day_mask, "quantidade"].sum())
            target_quantity = None
            if kind == "unit_value_rise":
                modified.loc[day_mask, "preco_unitario"] *= factor
                modified.loc[day_mask, "faturamento"] *= factor
            else:
                old_quantity = modified.loc[day_mask, "quantidade"].copy()
                new_quantity, target_quantity = allocate_integer_quantities(old_quantity, factor, rng)
                ratio = new_quantity.div(old_quantity.where(old_quantity.ne(0))).fillna(0)
                modified.loc[day_mask, "quantidade"] = new_quantity
                modified.loc[day_mask, "faturamento"] *= ratio
            after = float(modified.loc[day_mask, "faturamento"].sum())
            after_quantity = float(modified.loc[day_mask, "quantidade"].sum())
            effects.append({"data": pd.Timestamp(date).date().isoformat(),
                            "original_quantity": before_quantity, "target_quantity": target_quantity,
                            "realized_quantity": after_quantity,
                            "original_revenue": before, "realized_revenue": after,
                            "quantity_change_pct": ((after_quantity / before_quantity - 1) * 100 if before_quantity else None),
                            "revenue_change_pct": ((after / before - 1) * 100 if before else None)})
            if not np.isclose(before, after, rtol=0, atol=1e-8):
                label_dates[pd.Timestamp(date)] = event_id
        events.append({"event_id": event_id, "type": kind, "scope": scope, "target": target,
                       "start": str(start.date()), "end": str(end.date()), "factor": factor,
                       "daily_effects": effects,
                       "status": "completed" if any(value == event_id for value in label_dates.values()) else "no_effect"})
    return modified, events, label_dates


def _metrics(points, complete):
    evaluated = points.loc[points["evaluation_mask"]]
    positive = evaluated["is_anomaly"].eq(True)
    flag = evaluated["flag"].astype(bool)
    tp = int((positive & flag).sum())
    fn = int((positive & ~flag).sum())
    result = {"evaluated_points": len(evaluated), "known_positive_points": int(positive.sum()),
              "tp": tp, "fn": fn, "precision": None, "recall": None, "f1": None,
              "fp": None, "tn": None, "injection_recall": tp / (tp + fn) if tp + fn else None,
              "unlabeled_alerts": int((~positive & flag).sum()) if not complete else 0}
    if complete:
        fp = int((~positive & flag).sum())
        tn = int((~positive & ~flag).sum())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        result.update(fp=fp, tn=tn, precision=precision, recall=recall,
                      f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    return result


def evaluate_stage(data, stage="controlled", ground_truth_complete=True, seed=42):
    original = build_time_series(data)
    partitioned = temporal_split(original)
    _, calibration = _fit(partitioned)
    _, legacy_calibration = _fit(partitioned, method="quantile")
    modified, events, labels = _inject(data, partitioned, seed=seed)
    changed_series = build_time_series(modified)
    changed_series["split"] = partitioned["split"]
    robust = apply_threshold(score_historical_series(changed_series, trend="linear"), calibration)
    legacy = apply_threshold(robust, legacy_calibration)
    fixed = _baseline(changed_series)
    common = robust["eligible"] & fixed["eligible"] & robust["split"].eq("test")
    frames, metrics = [], []
    for name, frame in [("previous_week_30pct", fixed), ("historical_quantile98", legacy), ("seasonal_trend_mad", robust)]:
        frame = frame.copy()
        frame["stage"], frame["detector"] = stage, name
        frame["evaluation_mask"] = common
        frame["event_id"] = frame["data"].map(labels)
        positive = frame["event_id"].notna()
        frame["is_anomaly"] = positive if ground_truth_complete else positive.map({True: True, False: None})
        frame["label_status"] = np.where(positive, "injected_positive",
                                        "controlled_normal" if ground_truth_complete else "unlabeled_real")
        metrics.append({"stage": stage, "detector": name, **_metrics(frame, ground_truth_complete)})
        for event in events:
            mask = frame["event_id"].eq(event["event_id"]) & common
            event[name + "_detected"] = bool(frame.loc[mask, "flag"].any())
            event[name + "_eligible_points"] = int(mask.sum())
        frames.append(frame)
    digest = hashlib.sha256(data.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()
    summary = {
        "ground_truth_complete": ground_truth_complete, "input_rows": len(data),
        "input_sha256": digest, "splits": partitioned.attrs["split_boundaries"],
        "source_context": clean_json(data.attrs.get("dataset_context", {})),
        "calibration": calibration, "metrics": metrics, "events": events,
        "legacy_calibration": legacy_calibration,
        "eligible_test_points": int(common.sum()),
        "protocol": "Prequential: referências usam somente observações passadas; limiar congelado antes do teste.",
        "label_definition": ("Normalidade definida pelo gerador controlado, com eventos artificiais no teste."
                             if ground_truth_complete else
                             "Somente pontos injetados têm rótulo positivo; demais pontos reais permanecem sem rótulo."),
    }
    return summary, pd.concat(frames, ignore_index=True)


def run_anomaly_benchmark(output_dir="outputs/benchmark", seed=42, days=420, *,
                          real_data=None, olist_dir=None,
                          olist_start_period="2017-01", olist_end_period="2018-07"):
    stages = {}
    frames = []
    stages["controlled"], frame = evaluate_stage(controlled_sales(seed, days), seed=seed)
    frames.append(frame)
    if real_data is None and olist_dir is not None:
        from src.data.adapters.olist_adapter import OlistAdapter
        real_data = OlistAdapter(olist_dir).load()
        months = real_data["data"].dt.to_period("M").astype(str)
        real_data = real_data.loc[months.between(olist_start_period, olist_end_period)].copy()
    if real_data is not None:
        stages["olist_semisynthetic"], frame = evaluate_stage(
            real_data, "olist_semisynthetic", ground_truth_complete=False, seed=seed)
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    result = {"version": 1, "seed": seed, "days": days, "mode": "offline_no_llm",
              "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
              "configuration": {
                  "frequency": "D", "missing_policy": "unknown",
                  "train_fraction": .5, "calibration_fraction": .25,
                  "historical": {"seasonal_period": 7, "window": 8, "min_history": 4,
                                 "trend": "linear", "scale_floor_pct": .05, "absolute_floor": 1,
                                 "calibration_method": "robust", "robust_multiplier": 3,
                                 "legacy_calibration_quantile": .98, "minimum_threshold": 3},
                  "baseline": {"lag_days": 7, "threshold_pct": 30},
                  "quantity_allocation": "daily_total_largest_remainders_seeded_ties",
              },
              "stages": stages, "limitations": [
                  "Experimento demonstrativo com uma semente; não estabelece superioridade geral.",
                  "Métricas controladas dependem do gerador e da intensidade dos eventos.",
                  "Na Olist, recall de injeções mede somente os positivos artificiais conhecidos.",
                  "Alertas reais sem rótulo não são falsos positivos nem anomalias confirmadas.",
                  "O teste é retrospectivo; pedidos com status final entregue não simulam disponibilidade em tempo real.",
                  "Calendário e rótulos são diários; eventos de segmentos podem ser diluídos no faturamento total.",
              ]}
    real_stage = stages.get("olist_semisynthetic", {})
    for item in real_stage.get("metrics", []):
        if item["known_positive_points"] and item["tp"] == 0:
            result["limitations"].append(
                f'{item["detector"]} não detectou nenhuma injeção avaliável nesta execução. '
                "A calibração pode produzir um limiar conservador em séries voláteis; "
                "revisões de parâmetros exigem outra validação independente.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    points.drop(columns=["data"], errors="ignore").to_csv(out / "point_results.csv", index=False)
    pd.DataFrame([m for s in stages.values() for m in s["metrics"]]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame([{"stage": name, **e} for name, s in stages.items() for e in s["events"]]).to_csv(out / "events.csv", index=False)
    review = points.loc[points["label_status"].eq("unlabeled_real") & points["evaluation_mask"] & points["flag"],
                        ["stage", "detector", "periodo", "valor", "expected", "lower", "upper"]].copy()
    review["judgement"], review["notes"] = "", ""
    review.to_csv(out / "manual_review.csv", index=False)
    text = ["# Avaliação dos detectores", "", f"Semente: {seed}. Execução local, sem chamadas à IA.",
            "", "Protocolo: treino 50%, calibração 25%, teste restante, em ordem temporal.",
            "Referência diária: mesmo dia da semana nas 8 semanas anteriores; mínimo de 4 observações.",
            "Tendência: mediana das inclinações entre pares passados; dispersão MAD dos resíduos.",
            "Limiar robusto: máximo entre 3 e mediana + 3 × MAD escalado dos escores da calibração.",
            "Legado para comparação: máximo entre 3 e quantil 98% da calibração.",
            "Piso de escala: máximo entre MAD × 1,4826, 5% da referência e 1 unidade.",
            "Baseline: diferença absoluta superior a 30% contra o mesmo dia da semana anterior.",
            "Os três métodos usam a mesma máscara de pontos elegíveis no teste.",
            "A referência é atualizada com dados anteriores do teste; o limiar permanece congelado."]
    text.append("Volume: o total diário é arredondado uma vez e distribuído por maiores restos; preços preservados. Efeitos realizados ficam registrados nos eventos.")
    for name, stage in stages.items():
        text.extend(["", "## " + name, stage["label_definition"], "",
                     "Partições: " + json.dumps(stage["splits"], ensure_ascii=False)])
        for m in stage["metrics"]:
            text.append("- " + m["detector"] + ": " + json.dumps(clean_json(m), ensure_ascii=False))
    text.extend(["", "## Limitações", *["- " + item for item in result["limitations"]]])
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    result["outputs"] = {name: str(out / name) for name in [
        "report.md", "summary.json", "point_results.csv", "metrics.csv", "events.csv", "manual_review.csv"]}
    result = clean_json(result)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result

