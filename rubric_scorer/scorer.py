"""Deterministic scoring of supplied pass/fail judgments on a 0–100 scale."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

from .criteria import CriteriaSet, Criterion
from .hallucination import HallucinationAnnotation, summarize_hallucinations


@dataclass
class ScoreResult:
    response_id: str
    raw_score: float
    max_score: float
    normalized_score: float
    hard_failed: bool
    passed_criteria: list[str] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hallucination_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RubricScorer:
    """Apply a rubric to existing judgments; no model or network calls.

    A true judgment contributes its criterion's weight, including negative
    weights. Missing judgments count as false and are explicitly noted.
    Scores are divided by the sum of positive weights and clipped to 0–100.
    Failed hard-fail criteria and critical supplied annotations force zero.
    Criterion predicates are not executed by this scoring API.
    """

    def __init__(self, criteria_set: CriteriaSet):
        if not isinstance(criteria_set, CriteriaSet):
            raise TypeError("Expected a CriteriaSet instance.")
        self.rubric_name = criteria_set.name
        self.criteria = tuple(criteria_set.criteria)
        if not self.criteria:
            raise ValueError("A rubric must contain at least one criterion.")
        if any(not isinstance(c, Criterion) for c in self.criteria):
            raise TypeError("A rubric may contain only Criterion instances.")
        self._ids = {c.id for c in self.criteria}
        if len(self._ids) != len(self.criteria):
            raise ValueError("Criterion ids must be unique.")
        self.max_score = float(sum(c.weight for c in self.criteria if c.weight > 0))
        if self.max_score <= 0:
            raise ValueError("A rubric needs at least one positive-weight criterion.")

    def score(
        self,
        response_id: str,
        results: Mapping[str, bool],
        hallucinations: Iterable[HallucinationAnnotation | Mapping] | None = None,
    ) -> ScoreResult:
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("response_id must be a non-empty string.")
        if not isinstance(results, Mapping):
            raise TypeError("results must be an object of criterion ids and booleans.")
        if any(not isinstance(key, str) for key in results):
            raise TypeError("Evaluation keys must be criterion id strings.")
        unknown = set(results) - self._ids
        if unknown:
            raise ValueError(f"Unknown criterion ids: {', '.join(sorted(unknown))}.")
        if any(not isinstance(value, bool) for value in results.values()):
            raise TypeError("Evaluation values must be booleans, not strings or numbers.")

        passed, failed, notes = [], [], []
        raw_score = 0.0
        hard_failed = False
        for criterion in self.criteria:
            if criterion.id not in results:
                notes.append(f"Missing evaluation for {criterion.id!r}; treated as failed.")
            if results.get(criterion.id, False):
                passed.append(criterion.id)
                raw_score += criterion.weight
            else:
                failed.append(criterion.id)
                if criterion.hard_fail:
                    hard_failed = True

        items = [
            item if isinstance(item, HallucinationAnnotation)
            else HallucinationAnnotation.from_dict(item)
            for item in (hallucinations if hallucinations is not None else ())
        ]
        summary = summarize_hallucinations(items)
        if hard_failed:
            notes.append("Hard-fail criterion triggered — normalized score forced to 0.")
        if summary["has_critical"]:
            hard_failed = True
            notes.append("Critical hallucination annotation — normalized score forced to 0.")
        normalized = max(0.0, min(100.0, 100.0 * raw_score / self.max_score))
        if hard_failed:
            normalized = 0.0
        return ScoreResult(
            response_id=response_id,
            raw_score=raw_score,
            max_score=self.max_score,
            normalized_score=normalized,
            hard_failed=hard_failed,
            passed_criteria=passed,
            failed_criteria=failed,
            notes=notes,
            hallucination_summary=summary,
        )

    def score_batch(self, responses: Iterable[Mapping]) -> list[ScoreResult]:
        output = []
        for response in responses:
            if not isinstance(response, Mapping):
                raise TypeError("Each response must be an object.")
            if "response_id" not in response or "results" not in response:
                raise ValueError("Each response requires response_id and results.")
            output.append(self.score(
                response["response_id"], response["results"],
                hallucinations=response.get("hallucinations"),
            ))
        return output
