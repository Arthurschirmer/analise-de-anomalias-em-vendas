"""Offline comparison with known synthetic context and a separate answer key.

Only unexpected injected changes are positives in this conditional task.
Promotion effects prescribed by the generator are expected negatives;
this convention does not claim those effects are normal in every business.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.contextual_detection import analyze_contextual_series
from src.analysis.historical_detection import (
    apply_threshold, build_time_series, calibrate_threshold,
    score_historical_series, temporal_split,
)
from src.data.contextual_demo import generate_contextual_sales
from src.data.loader import prepare_sales_data
from src.evaluation.anomaly_benchmark import clean_json


DETECTORS = ("historical_without_context", "contextual")
METRIC_FIELDS = (
    "evaluated_points", "known_positive_points", "tp", "fp", "fn", "tn",
    "precision", "recall", "f1", "false_alert_rate", "alert_count",
    "expected_context_points", "expected_context_alerts", "expected_context_alert_rate",
    "context_positive_points", "context_positive_detected",
    "ordinary_positive_points", "ordinary_positive_detected",
)
POINT_COLUMNS = [
    "data", "periodo", "valor", "split", "seed", "detector", "expected",
    "lower", "upper", "flag", "eligible", "evaluation_mask", "is_anomaly",
    "promocao", "context_known", "expected_context", "context_label",
]
LIMITATIONS = [
    "Todos os dados são fictícios. As promoções afetam as vendas porque essas regras foram definidas no gerador.",
    "O experimento não demonstra causalidade em uma empresa real nem superioridade geral do método contextual.",
    "Nesta tarefa, somente mudanças inesperadas são anomalias; efeitos previstos de promoção são negativos conhecidos.",
    "O modelo contextual recebe somente vendas e calendário. O gabarito é usado depois das previsões, apenas na avaliação.",
    "O contexto é completo neste cenário. Dados externos ausentes ou incorretos podem impedir ou prejudicar a análise real.",
    "Os métodos têm referências diferentes: o histórico atualiza sua janela com o passado, enquanto o modelo contextual permanece ajustado ao treino.",
    "Médias e desvios entre sementes não constituem intervalos de confiança nem uma validação independente em dados reais.",
]


def _hash_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")).hexdigest()


def _annotations(series, calendar, truth):
    # This is called only after both detectors have produced their predictions.
    labels = truth.copy()
    if not {"data", "is_anomaly"}.issubset(labels.columns):
        raise ValueError("O gabarito precisa de data e is_anomaly.")
    labels["data"] = pd.to_datetime(labels["data"], errors="raise").dt.normalize()
    if labels["data"].duplicated().any() or labels["is_anomaly"].isna().any():
        raise ValueError("O gabarito precisa de datas únicas e classificações completas.")
    if not labels["is_anomaly"].isin([True, False]).all():
        raise ValueError("O gabarito deve usar classificações booleanas.")
    columns = ["data", "is_anomaly"] + (["event_id"] if "event_id" in labels else [])
    result = series[["data"]].merge(labels[columns], on="data", how="left", validate="one_to_one")
    if result["is_anomaly"].isna().any():
        raise ValueError("O gabarito não cobre todas as datas da série.")
    context = calendar[["data", "promocao"]].copy()
    context["data"] = pd.to_datetime(context["data"], errors="raise").dt.normalize()
    result = result.merge(context, on="data", how="left", validate="one_to_one")
    result["is_anomaly"] = result["is_anomaly"].astype(bool)
    result["context_known"] = result["promocao"].notna()
    promotion = result["promocao"].astype("boolean").fillna(False)
    result["expected_context"] = promotion & result["context_known"]
    result["context_label"] = np.select(
        [~result["context_known"], promotion.to_numpy(dtype=bool)],
        ["unknown", "promotion"], default="normal",
    )
    return result


def _metrics(frame):
    evaluated = frame.loc[frame["evaluation_mask"]]
    positive = evaluated["is_anomaly"].astype(bool)
    flag = evaluated["flag"].astype(bool)
    context = evaluated["expected_context"].astype(bool)
    tp, fp = int((positive & flag).sum()), int((~positive & flag).sum())
    fn, tn = int((positive & ~flag).sum()), int((~positive & ~flag).sum())
    expected = context & ~positive
    expected_count = int(expected.sum())
    expected_alerts = int((expected & flag).sum())
    return {
        "evaluated_points": len(evaluated), "known_positive_points": int(positive.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_alert_rate": fp / (fp + tn) if fp + tn else None,
        "alert_count": tp + fp,
        "expected_context_points": expected_count,
        "expected_context_alerts": expected_alerts,
        "expected_context_alert_rate": expected_alerts / expected_count if expected_count else None,
        "context_positive_points": int((context & positive).sum()),
        "context_positive_detected": int((context & positive & flag).sum()),
        "ordinary_positive_points": int((~context & positive).sum()),
        "ordinary_positive_detected": int((~context & positive & flag).sum()),
    }


def evaluate_contextual_dataset(sales, calendar, truth, *, seed=42):
    """Predict without labels, then compare both methods on identical test days."""
    series = temporal_split(build_time_series(sales))
    historical = score_historical_series(series, trend="linear")
    historical_calibration = calibrate_threshold(historical, method="robust")
    historical = apply_threshold(historical, historical_calibration)
    contextual = analyze_contextual_series(
        series, calendar, min_train=56, min_context_occurrences=3,
        scale_floor_pct=.05, absolute_floor=1.0, minimum_threshold=3.0,
    )
    summary = {
        "seed": seed, "available": bool(contextual["available"]),
        "reason": contextual.get("reason"), "splits": series.attrs["split_boundaries"],
        "historical_calibration": historical_calibration,
        "contextual_calibration": contextual.get("calibration"),
        "contextual_model": contextual.get("model"), "metrics": [],
        "limitations": contextual.get("limitations", []),
    }
    if not contextual["available"]:
        return clean_json(summary), pd.DataFrame(columns=POINT_COLUMNS)
    contextual_points = contextual["scored"]
    if not contextual_points["data"].reset_index(drop=True).equals(series["data"]):
        raise ValueError("Os detectores precisam preservar o mesmo calendário.")
    common = historical["eligible"].fillna(False) & contextual_points["eligible"].fillna(False) & series["split"].eq("test")
    annotations = _annotations(series, calendar, truth)
    summary["eligible_test_points"] = int(common.sum())
    summary["ineligible_test_points"] = int((series["split"].eq("test") & ~common).sum())
    frames = []
    for name, scored in zip(DETECTORS, (historical, contextual_points)):
        frame = scored.drop(columns=["promocao"], errors="ignore").copy()
        frame = frame.merge(annotations, on="data", how="left", validate="one_to_one")
        frame["seed"], frame["detector"] = seed, name
        frame["evaluation_mask"] = common.to_numpy(dtype=bool)
        summary["metrics"].append({"seed": seed, "detector": name, **_metrics(frame)})
        frames.append(frame)
    return clean_json(summary), pd.concat(frames, ignore_index=True)


def _aggregate(metrics):
    frame = pd.DataFrame(metrics)
    result = []
    if frame.empty:
        return result
    for detector, part in frame.groupby("detector", sort=True):
        row = {"detector": detector, "runs": len(part)}
        for name in METRIC_FIELDS:
            values = pd.to_numeric(part[name], errors="coerce").dropna()
            row[name + "_n"] = len(values)
            row[name + "_mean"] = float(values.mean()) if len(values) else None
            row[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
        result.append(row)
    return result


def _report(result):
    source_description = (
        "Base fictícia carregada dos arquivos locais, sem regenerar as vendas ou o calendário. Identificador da demonstração: 42."
        if result["mode"] == "offline_synthetic_fixture"
        else "Sementes: " + ", ".join(map(str, result["seeds"]))
    )
    results_heading = ("## Resultado da base carregada" if result["mode"] == "offline_synthetic_fixture"
                       else "## Resultados médios entre sementes")
    lines = ["# Experimento fictício: promoções e quedas inesperadas", "", source_description,
             "Execução local, sem chamadas à IA. Treino 50%, calibração 25% e teste 25%, em ordem de data.", "",
             "Promoções fazem parte do comportamento esperado definido neste cenário. Só as quedas-surpresa adicionais são anomalias.",
             "O detector sem contexto recebe as vendas. O contextual recebe as mesmas vendas e o calendário de promoções.",
             "O gabarito é consultado somente depois das previsões. Ambos são avaliados nos mesmos dias elegíveis do teste.", "",
             results_heading, "",
             "| Método | Execuções | Precisão | Recuperação das surpresas | F1 | Alertas em dias de contexto esperado |",
             "|---|---:|---:|---:|---:|---:|"]
    labels = {"historical_without_context": "Histórico sem contexto", "contextual": "Contextual"}
    for row in result["aggregate_metrics"]:
        def number(name):
            value = row.get(name + "_mean")
            return "não calculável" if value is None else f"{value:.3f}"
        lines.append(f"| {labels[row['detector']]} | {row['runs']} | {number('precision')} | {number('recall')} | {number('f1')} | {number('expected_context_alerts')} |")
    unavailable = [run for run in result["runs"] if not run["available"]]
    if unavailable:
        lines.extend(["", "## Comparações indisponíveis", *[f"- Semente {run['seed']}: {run['reason']}" for run in unavailable]])
    snapshot_description = (
        "vendas.csv, calendario.csv e gabarito.csv são cópias padronizadas dos arquivos lidos; não foram regenerados."
        if result["mode"] == "offline_synthetic_fixture"
        else "vendas.csv, calendario.csv e gabarito.csv são separados e representam a primeira semente da lista."
    )
    lines.extend(["", "Precisão é a proporção de alertas corretos. Recuperação indica quantas surpresas conhecidas foram encontradas. F1 combina essas duas medidas.",
                  "Alertas nos dias de contexto esperado são contados somente quando não existe uma surpresa adicional naquele dia.",
                  "As colunas *_n dos arquivos de métricas indicam quantas execuções contribuíram para cada média.",
                  "", "## Limitações", *["- " + value for value in result["limitations"]],
                  "", "## Arquivos", "",
                  snapshot_description,
                  "point_results.csv contém as previsões de todas as sementes; metrics.csv e aggregate_metrics.csv guardam a avaliação."])
    return "\n".join(lines) + "\n"


def _write_results(out, *, runs, frames, seeds, days,
                   mode="offline_synthetic_context", source=None, limitations=None):
    """Export the shared comparison format after predictions and validation."""
    metrics = [item for run in runs for item in run["metrics"]]
    points = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=POINT_COLUMNS)
    points.to_csv(out / "point_results.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(metrics, columns=["seed", "detector", *METRIC_FIELDS]).to_csv(out / "metrics.csv", index=False)
    aggregated = _aggregate(metrics)
    pd.DataFrame(aggregated, columns=["detector", "runs", *[name + suffix for name in METRIC_FIELDS for suffix in ("_n", "_mean", "_std")]]).to_csv(out / "aggregate_metrics.csv", index=False)
    names = ("report.md", "summary.json", "point_results.csv", "metrics.csv", "aggregate_metrics.csv", "vendas.csv", "calendario.csv", "gabarito.csv")
    result = {
        "version": 2, "context_columns": ["promocao"], "mode": mode, "seeds": seeds, "days": days,
        "snapshot_seed": seeds[0], "runs": runs, "metrics": metrics, "aggregate_metrics": aggregated,
        "available_runs": sum(run["available"] for run in runs),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "protocol": {"splits": "chronological_50_25_25", "labels_used_for_fit": False,
                     "contextual_fit": "train_only", "threshold_fit": "calibration_only",
                     "evaluation": "shared_eligible_test_mask", "positive_definition": "unexpected_injected_changes_only",
                     "historical_trend": "linear", "historical_calibration": "robust",
                     "contextual_configuration": {"min_train": 56, "min_context_occurrences": 3, "scale_floor_pct": .05, "absolute_floor": 1.0, "minimum_threshold": 3.0}},
        "limitations": list(LIMITATIONS if limitations is None else limitations),
        "outputs": {name: str(out / name) for name in names},
    }
    if source is not None:
        result.update(source=source, fixture_seed=42)
    result = clean_json(result)
    (out / "report.md").write_text(_report(result), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def _fixture_boolean(values, column):
    """Preserve unknown values instead of interpreting nonempty strings as true."""
    def parse(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
        elif isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)) and value in (0, 1):
            return bool(value)
        raise ValueError(f"A coluna {column} deve usar True/False, 0/1 ou valores ausentes.")
    return values.map(parse).astype("boolean")


def _validate_fixture_revenue(sales, truth):
    """Check answer-key provenance after prediction; never use it to fit a model."""
    if "revenue" not in truth:
        return False
    observed = sales.assign(data=sales["data"].dt.normalize()).groupby("data", as_index=False)["faturamento"].sum()
    recorded = truth[["data", "revenue"]].copy()
    recorded["revenue"] = pd.to_numeric(recorded["revenue"], errors="coerce")
    if recorded["data"].duplicated().any():
        raise ValueError("O gabarito precisa de datas únicas.")
    paired = observed.merge(recorded, on="data", how="outer", validate="one_to_one")
    values = paired[["faturamento", "revenue"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values[:, 0], values[:, 1], rtol=1e-10, atol=1e-6):
        raise ValueError("O faturamento diário não corresponde ao gabarito da base fictícia.")
    return True


def run_contextual_fixture(input_dir, output_dir="outputs/contextual/loaded_demo"):
    """Evaluate the existing synthetic CSV files without regenerating their data.

    The run identifier 42 refers to the demonstration fixture, not to a fresh
    random generation. Raw input hashes and prepared-data hashes identify the
    actual files evaluated. The independent calendar may contain unknown dates.
    """
    folder, out = Path(input_dir), Path(output_dir)
    if folder.resolve() == out.resolve():
        raise ValueError("A pasta de resultados deve ser diferente da pasta dos arquivos de entrada.")
    paths = {name: folder / name for name in ("vendas.csv", "calendario.csv", "gabarito.csv")}
    sales = prepare_sales_data(pd.read_csv(paths["vendas.csv"]))
    if sales.empty or sales["data"].isna().any():
        raise ValueError("O arquivo de vendas precisa de registros com datas válidas.")
    numeric = sales[["quantidade", "preco_unitario", "desconto", "faturamento"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("O arquivo de vendas contém números ausentes ou inválidos.")
    calendar = pd.read_csv(paths["calendario.csv"])
    if not {"data", "promocao"}.issubset(calendar):
        raise ValueError("O calendário precisa de data e promocao.")
    calendar = calendar[["data", "promocao"]].copy()
    calendar["data"] = pd.to_datetime(calendar["data"], errors="raise").dt.normalize()
    for column in ("promocao",):
        calendar[column] = _fixture_boolean(calendar[column], column)
    truth = pd.read_csv(paths["gabarito.csv"])
    if not {"data", "is_anomaly"}.issubset(truth):
        raise ValueError("O gabarito precisa de data e is_anomaly.")
    truth["data"] = pd.to_datetime(truth["data"], errors="raise").dt.normalize()
    truth["is_anomaly"] = _fixture_boolean(truth["is_anomaly"], "is_anomaly")

    run, points = evaluate_contextual_dataset(sales, calendar, truth, seed=42)
    reconciled = _validate_fixture_revenue(sales, truth)
    run["input_hashes"] = {"sales": _hash_frame(sales), "calendar": _hash_frame(calendar), "truth": _hash_frame(truth)}
    source = {
        "kind": "local_synthetic_fixture", "regenerated": False,
        "run_identifier": 42, "identifier_note": "Identificador da demonstração; os arquivos não foram regenerados.",
        "input_dir": str(folder.resolve()), "truth_revenue_reconciled": reconciled,
        "files": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name, path in paths.items()},
    }
    run["source"] = source
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("vendas.csv", sales), ("calendario.csv", calendar), ("gabarito.csv", truth)):
        frame.to_csv(out / name, index=False)
    limitations = []
    for value in LIMITATIONS:
        if value.startswith("O contexto é completo"):
            value = "O calendário é o informado nos arquivos; contexto ausente não é preenchido como ausência de promoção."
        elif value.startswith("Médias e desvios entre sementes"):
            value = "Uma base fictícia carregada foi avaliada; esse resultado não constitui validação independente em dados reais."
        limitations.append(value)
    return _write_results(out, runs=[run], frames=[points] if not points.empty else [],
                          seeds=[42], days=sum(item["count"] for item in run["splits"].values()),
                          mode="offline_synthetic_fixture", source=source, limitations=limitations)


def run_contextual_benchmark(output_dir="outputs/contextual", seeds=(42, 137, 2026), days=420):
    """Generate, evaluate and export a fixed illustrative experiment."""
    seeds = list(seeds)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds):
        raise ValueError("Informe uma ou mais sementes inteiras não negativas.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("As sementes da comparação precisam ser distintas.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, frames = [], []
    for index, seed in enumerate(seeds):
        dataset = generate_contextual_sales(seed=int(seed), days=days)
        run, points = evaluate_contextual_dataset(dataset.sales, dataset.calendar, dataset.truth, seed=int(seed))
        run["input_hashes"] = {"sales": _hash_frame(dataset.sales), "calendar": _hash_frame(dataset.calendar), "truth": _hash_frame(dataset.truth)}
        run["generator"] = clean_json(dataset.metadata)
        runs.append(run)
        if not points.empty:
            frames.append(points)
        if index == 0:
            for name, frame in [("vendas.csv", dataset.sales), ("calendario.csv", dataset.calendar), ("gabarito.csv", dataset.truth)]:
                frame.to_csv(out / name, index=False, date_format="%Y-%m-%d")
    return _write_results(out, runs=runs, frames=frames, seeds=seeds, days=days)
