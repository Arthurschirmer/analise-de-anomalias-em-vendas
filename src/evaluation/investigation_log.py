import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class InvestigationLog:
    anomaly_description: str
    final_conclusion: str
    tool_results: list
    evidence: dict
    structured_report: dict
    evaluation: dict | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        data = {
            "created_at": self.created_at,
            "anomaly_description": self.anomaly_description,
            "tools_used": [tool_result.to_dict() for tool_result in self.tool_results],
            "evidence": self.evidence,
            "structured_report": self.structured_report,
            "final_conclusion": self.final_conclusion,
        }

        if self.evaluation is not None:
            data["evaluation"] = self.evaluation

        return data

    @classmethod
    def from_dict(cls, record):
        """Restore stored results without running tools or a language model."""
        from src.tools.sales_tools import ToolResult
        if not isinstance(record, dict) or not isinstance(record.get("structured_report"), dict):
            raise ValueError("Registro de investigação inválido.")
        if any(not isinstance(record.get(key), str) for key in ("anomaly_description", "final_conclusion", "created_at")):
            raise ValueError("Os campos de identificação da investigação são inválidos.")
        datetime.fromisoformat(record["created_at"])
        if not isinstance(record.get("tools_used", []), list) or not isinstance(record.get("evidence", {}), dict):
            raise ValueError("Evidências ou ferramentas armazenadas são inválidas.")
        if not isinstance(record["structured_report"].get("periodos", {}), dict):
            raise ValueError("Períodos armazenados são inválidos.")
        return cls(anomaly_description=record["anomaly_description"], final_conclusion=record["final_conclusion"],
                   tool_results=[ToolResult(name=item["name"], parameters=item["parameters"], result=item["result"])
                                 for item in record.get("tools_used", [])], evidence=record.get("evidence", {}),
                   structured_report=record["structured_report"], evaluation=record.get("evaluation"),
                   created_at=record["created_at"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
