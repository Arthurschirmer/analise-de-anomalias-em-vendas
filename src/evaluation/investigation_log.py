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

    def save(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
