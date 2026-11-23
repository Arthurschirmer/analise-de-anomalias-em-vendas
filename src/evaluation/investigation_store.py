"""Local, atomic investigation snapshots bound to data and business semantics."""

from datetime import date, datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

from src.data.loader import STANDARD_COLUMNS
from src.evaluation.investigation_log import InvestigationLog


SEMANTIC_FIELDS = ("currency", "discount_available", "available_dimensions", "category_available",
                   "seller_available", "region_available", "product_available", "revenue_definition",
                   "quantity_definition", "row_definition", "price_definition")


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def dataset_fingerprint(data):
    canonical = data[STANDARD_COLUMNS].copy()
    canonical["data"] = canonical["data"].astype("string")
    for column in ("quantidade", "preco_unitario", "desconto", "faturamento"):
        canonical[column] = canonical[column].astype("float64")
    for column in ("produto", "categoria", "regiao", "vendedor"):
        canonical[column] = canonical[column].astype("string")
    digest = sha256(pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes())
    context = data.attrs.get("dataset_context", {})
    # Defaults make a reload of an ordinary CSV stable across UI enrichment.
    semantics = {key: context[key] for key in SEMANTIC_FIELDS if key in context}
    digest.update(json.dumps(_json_value(semantics), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def save_investigation(log, data, directory, *, label="Base de vendas", mode="local", feedback=None):
    if mode not in {"local", "ai"}:
        raise ValueError("Modo de investigação inválido.")
    entry_id = uuid4().hex
    record = _json_value({"version": 1, "id": entry_id, "dataset_sha256": dataset_fingerprint(data),
        "dataset_label": str(label), "mode": mode, "feedback": feedback,
        "record": log.to_dict()})
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (entry_id + ".tmp")
    destination = output / (entry_id + ".json")
    temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return record


def load_investigation(directory, entry_id, expected_dataset=None):
    if not isinstance(entry_id, str) or not re.fullmatch(r"[a-f0-9]{32}", entry_id):
        raise ValueError("Identificador da investigação inválido.")
    path = Path(directory) / (entry_id + ".json")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("O registro excede o limite de leitura.")
    entry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict) or entry.get("version") != 1 or entry.get("id") != entry_id:
        raise ValueError("O formato do registro não é reconhecido.")
    if not isinstance(entry.get("dataset_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", entry["dataset_sha256"]):
        raise ValueError("A identificação da base está inválida no registro.")
    if entry.get("mode") not in {"local", "ai"} or not isinstance(entry.get("dataset_label"), str):
        raise ValueError("Os dados de apresentação do registro são inválidos.")
    if expected_dataset is not None and entry.get("dataset_sha256") != expected_dataset:
        raise ValueError("Esta investigação pertence a outra base ou a outras definições. Carregue a mesma base para reabri-la.")
    InvestigationLog.from_dict(entry["record"])
    return entry


def list_investigations(directory, dataset_hash=None):
    entries, invalid = [], 0
    for path in Path(directory).glob("*.json"):
        if not re.fullmatch(r"[a-f0-9]{32}", path.stem):
            continue
        try:
            entry = load_investigation(directory, path.stem)
        except (OSError, ValueError, KeyError, TypeError):
            invalid += 1
            continue
        if dataset_hash is None or entry["dataset_sha256"] == dataset_hash:
            entries.append(entry)
    entries.sort(key=lambda item: item["record"]["created_at"], reverse=True)
    return entries, invalid
