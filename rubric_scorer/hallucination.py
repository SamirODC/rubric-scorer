"""Explicit human-supplied hallucination labels, not automatic fact checking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class HallucinationCategory(str, Enum):
    EXPLORATION = "exploration"
    GROUNDING = "grounding"
    EXTERNAL_FACTS = "external_facts"
    INCORRECT_ACTION = "incorrect_action"


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass(frozen=True)
class HallucinationAnnotation:
    category: HallucinationCategory
    severity: Severity
    span: str
    explanation: str
    step_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", HallucinationCategory(self.category))
        object.__setattr__(self, "severity", Severity(self.severity))
        if not isinstance(self.span, str) or not isinstance(self.explanation, str):
            raise TypeError("Annotation span and explanation must be strings.")
        if self.step_index is not None and (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer or None.")

    @classmethod
    def from_dict(cls, value: Mapping) -> "HallucinationAnnotation":
        if not isinstance(value, Mapping):
            raise TypeError("Each hallucination annotation must be an object.")
        return cls(**value)


def summarize_hallucinations(
    annotations: Iterable[HallucinationAnnotation],
) -> dict:
    items = list(annotations)
    if any(not isinstance(a, HallucinationAnnotation) for a in items):
        raise TypeError("Expected HallucinationAnnotation instances.")
    categories = Counter(a.category.value for a in items)
    severities = Counter(a.severity.value for a in items)
    return {
        "total": len(items),
        "by_category": {c.value: categories[c.value] for c in HallucinationCategory},
        "by_severity": {s.value: severities[s.value] for s in Severity},
        "has_critical": severities[Severity.CRITICAL.value] > 0,
    }
