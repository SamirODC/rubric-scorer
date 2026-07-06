"""
Criteria definitions for rubric-based scoring.

Design principle (learned from production RLHF annotation work): a good
rubric criterion is ATOMIC (tests exactly one thing), OUTCOME-ORIENTED
(judges the final answer, not just the reasoning trace), and BINARY
(pass/fail — no partial credit that annotators interpret differently).

A healthy criteria set keeps roughly an 80/20 split between OUTCOME and
TRACE criteria, so annotators aren't rewarding "showed good work" over
"got the right answer." `CriteriaSet.outcome_trace_ratio()` and
`CriteriaSet.validate_ratio()` let you check that before you ship a rubric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class CriterionCategory(str, Enum):
    """Whether a criterion judges the final outcome or the reasoning trace."""

    OUTCOME = "outcome"
    TRACE = "trace"


@dataclass(frozen=True)
class Criterion:
    """A single, atomic, weighted grading criterion.

    Attributes:
        id: Short stable identifier, e.g. "no_fabricated_citation".
        description: Human-readable statement of what "pass" means.
        weight: Contribution to the total score if satisfied. Convention
            used throughout this library is -5..+5 (positive weights
            reward desired behavior, negative weights penalize it).
        category: OUTCOME or TRACE — see module docstring.
        hard_fail: If True, failing this criterion zeroes out the whole
            response regardless of other scores (e.g. safety violations,
            fabricated sources).
        predicate: Optional callable(response, context) -> bool that can
            auto-evaluate this criterion. If omitted, the criterion is
            expected to be scored manually (human annotation).
    """

    id: str
    description: str
    weight: float
    category: CriterionCategory = CriterionCategory.OUTCOME
    hard_fail: bool = False
    predicate: Optional[Callable[[str, Optional[dict]], bool]] = None

    def __post_init__(self) -> None:
        if not -5 <= self.weight <= 5:
            raise ValueError(
                f"Criterion '{self.id}' weight {self.weight} out of range "
                "(-5..+5)."
            )


@dataclass
class CriteriaSet:
    """A named collection of criteria that together form one rubric."""

    name: str
    criteria: list[Criterion] = field(default_factory=list)

    def add(self, criterion: Criterion) -> "CriteriaSet":
        self.criteria.append(criterion)
        return self

    def outcome_trace_ratio(self) -> tuple[float, float]:
        """Return (outcome_weight_share, trace_weight_share), each 0..1.

        Uses absolute weight as the unit of "importance" so a -5 hard
        fail counts as much as a +5 reward when computing the split.
        """
        outcome_w = sum(
            abs(c.weight) for c in self.criteria
            if c.category == CriterionCategory.OUTCOME
        )
        trace_w = sum(
            abs(c.weight) for c in self.criteria
            if c.category == CriterionCategory.TRACE
        )
        total = outcome_w + trace_w
        if total == 0:
            return (0.0, 0.0)
        return (outcome_w / total, trace_w / total)

    def validate_ratio(
        self, target_outcome_share: float = 0.8, tolerance: float = 0.15
    ) -> bool:
        """Check the set is roughly `target_outcome_share` outcome-weighted.

        Returns True if within tolerance, False otherwise. Doesn't raise,
        since some rubrics (e.g. pure process audits) intentionally break
        the 80/20 convention — this is a lint, not a hard rule.
        """
        outcome_share, _ = self.outcome_trace_ratio()
        return abs(outcome_share - target_outcome_share) <= tolerance

    def hard_fail_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.hard_fail]

    def max_possible_score(self) -> float:
        return sum(c.weight for c in self.criteria if c.weight > 0)

    def min_possible_score(self) -> float:
        return sum(c.weight for c in self.criteria if c.weight < 0)
