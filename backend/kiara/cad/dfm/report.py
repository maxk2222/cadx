"""DfM report model + assembly."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DfmWarning(BaseModel):
    code: str
    severity: str  # high | medium | low
    message: str
    value: float | None = None
    threshold: float | None = None
    location: str | None = None


class DfmReport(BaseModel):
    process: str
    warnings: list[DfmWarning] = Field(default_factory=list)
    summary: str = ""

    def sort(self) -> DfmReport:
        order = {"high": 0, "medium": 1, "low": 2}
        self.warnings.sort(key=lambda w: order.get(w.severity, 3))
        counts = {s: sum(1 for w in self.warnings if w.severity == s) for s in order}
        self.summary = (
            f"{counts['high']} high, {counts['medium']} medium, {counts['low']} low "
            f"(heuristic checks on tessellated geometry)"
        )
        return self
