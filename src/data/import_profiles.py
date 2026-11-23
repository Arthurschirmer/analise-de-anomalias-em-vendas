"""Local, versioned import settings. Profiles never contain sales records."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from src.data.loader import STANDARD_COLUMNS


PROFILE_DIR = Path(os.environ.get("ANOMALY_IMPORT_PROFILE_DIR") or Path(__file__).resolve().parents[2] / "outputs/import_profiles")
CHOICES = {
    "currency": ("BRL", "GBP", "USD", "EUR"),
    "date_format": ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y", "mixed"),
    "decimal": (".", ","), "revenue_mode": ("column", "calculate"),
    "price_mode": ("column", "calculate"), "invalid_policy": ("raise", "drop"),
}


def validate_profile_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("mapping"), dict):
        raise ValueError("Perfil sem mapeamento válido.")
    result = {}
    for key, options in CHOICES.items():
        if config.get(key) not in options:
            raise ValueError(f"Opção inválida no perfil: {key}.")
        result[key] = config[key]
    mapping = config["mapping"]
    if set(mapping) != set(STANDARD_COLUMNS) or any(value is not None and (not isinstance(value, str) or not value or len(value) > 500) for value in mapping.values()):
        raise ValueError("Colunas inválidas no perfil.")
    result["mapping"] = dict(mapping)
    required = ["data", "produto", "quantidade"]
    required += [field for field, mode in (("faturamento", "revenue_mode"), ("preco_unitario", "price_mode")) if result[mode] == "column"]
    if any(not mapping[field] for field in required) or result["revenue_mode"] == result["price_mode"] == "calculate":
        raise ValueError("Perfil sem as colunas necessárias ao cálculo.")
    selected = [value for value in mapping.values() if value is not None]
    if len(selected) != len(set(selected)):
        raise ValueError("O perfil associa a mesma coluna a mais de um campo.")
    if not isinstance(config.get("discount_percent"), bool):
        raise ValueError("Formato do desconto inválido.")
    result["discount_percent"] = config["discount_percent"]
    reading = config.get("read_options", {})
    if not isinstance(reading, dict):
        raise ValueError("Configuração de leitura inválida.")
    separator, encoding, sheet = reading.get("separator", ","), reading.get("encoding", "utf-8-sig"), reading.get("sheet")
    if separator not in (",", ";", "\t", "|") or encoding not in ("utf-8-sig", "latin-1") or (sheet is not None and (not isinstance(sheet, str) or not sheet or len(sheet) > 100)):
        raise ValueError("Configuração de leitura inválida.")
    result["read_options"] = {"separator": separator, "encoding": encoding, "sheet": sheet}
    return result


def missing_profile_columns(config, columns):
    return sorted({column for column in config["mapping"].values() if column is not None} - set(columns))


def save_import_profile(name, config, directory=None):
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError("Dê ao perfil um nome de 1 a 80 caracteres.")
    config = validate_profile_config(config)
    folder = Path(directory) if directory is not None else PROFILE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    record = {"version": 1, "id": uuid4().hex, "name": name.strip(), "created_at": datetime.now(timezone.utc).isoformat(), "config": config}
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, ensure_ascii=False, indent=2)
        temporary.replace(folder / (record["id"] + ".json"))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


def list_import_profiles(directory=None):
    folder = Path(directory) if directory is not None else PROFILE_DIR
    records, invalid = [], 0
    for path in folder.glob("*.json"):
        try:
            if path.stat().st_size > 65536:
                raise ValueError("Perfil muito grande.")
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("version") != 1 or not re.fullmatch(r"[0-9a-f]{32}", str(record.get("id", ""))) or path.stem != record["id"]:
                raise ValueError("Perfil inválido.")
            if not isinstance(record.get("name"), str) or not 1 <= len(record["name"].strip()) <= 80:
                raise ValueError("Nome inválido.")
            datetime.fromisoformat(record["created_at"])
            record["config"] = validate_profile_config(record["config"])
            records.append(record)
        except (OSError, ValueError, TypeError, KeyError):
            invalid += 1
    return sorted(records, key=lambda record: record["created_at"], reverse=True), invalid
