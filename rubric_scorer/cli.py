"""JSON input and CLI for reproducible, offline rubric scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .criteria import CriteriaSet, Criterion
from .scorer import RubricScorer, ScoreResult


def load_criteria_set(path: str | Path) -> CriteriaSet:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Criteria file must contain a JSON object.")
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError("Criteria file requires a non-empty name.")
    if not isinstance(data.get("criteria"), list):
        raise ValueError("Criteria file requires a criteria array.")
    criteria_set = CriteriaSet(name=data["name"])
    allowed = {"id", "description", "weight", "category", "hard_fail"}
    for index, item in enumerate(data["criteria"]):
        if not isinstance(item, dict):
            raise ValueError(f"Criterion {index} must be an object.")
        if set(item) - allowed:
            raise ValueError(f"Criterion {index} contains unsupported fields.")
        try:
            criteria_set.add(Criterion(**item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid criterion {index}: {exc}") from exc
    RubricScorer(criteria_set)  # Validate uniqueness, nonempty rubric and denominator.
    return criteria_set


def _print_scores(criteria_set: CriteriaSet, scores: list[ScoreResult]) -> None:
    outcome, trace = criteria_set.outcome_trace_ratio()
    ratio_status = "OK" if criteria_set.validate_ratio() else "outside target"
    print(f"Rubric: {criteria_set.name} ({len(criteria_set.criteria)} criteria)")
    print(f"Outcome/Trace weight split: {outcome:.0%} / {trace:.0%} ({ratio_status})")
    print("-" * 60)
    for result in scores:
        label = "HARD FAIL" if result.hard_failed else f"{result.normalized_score:.1f}/100"
        print(f"{result.response_id:24s} {label}")
        if result.failed_criteria:
            print(f"  failed: {', '.join(result.failed_criteria)}")
        for note in result.notes:
            print(f"  note: {note}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_parser = subparsers.add_parser("score", help="Score supplied JSON judgments.")
    score_parser.add_argument("--criteria", required=True)
    score_parser.add_argument("--responses", required=True)
    score_parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    validate_parser = subparsers.add_parser("validate-criteria", help="Validate a rubric and its ratio.")
    validate_parser.add_argument("--criteria", required=True)
    args = parser.parse_args(argv)
    try:
        criteria_set = load_criteria_set(args.criteria)
        if args.command == "validate-criteria":
            outcome, trace = criteria_set.outcome_trace_ratio()
            valid = criteria_set.validate_ratio()
            print(f"Rubric: {criteria_set.name} ({len(criteria_set.criteria)} criteria)")
            print(f"Outcome/Trace weight split: {outcome:.0%} / {trace:.0%}")
            print("OK" if valid else "Outcome/trace ratio is outside the configured target.")
            return 0 if valid else 1
        responses = json.loads(Path(args.responses).read_text(encoding="utf-8"))
        if not isinstance(responses, list):
            raise ValueError("Responses file must contain a JSON array.")
        scores = RubricScorer(criteria_set).score_batch(responses)
        if args.json:
            print(json.dumps([r.to_dict() for r in scores], ensure_ascii=False, indent=2))
        else:
            _print_scores(criteria_set, scores)
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
